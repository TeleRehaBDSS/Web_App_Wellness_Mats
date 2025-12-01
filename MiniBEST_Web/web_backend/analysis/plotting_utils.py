import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from PIL import Image

# Set non-interactive backend
plt.switch_backend('Agg')

# Set Dark Style for Plots
plt.style.use('dark_background')

def get_smooth_contour(contour):
    """
    Generates a smooth, tight polygon around the raw contour.
    """
    if contour is None or len(contour) < 3:
        return None
    
    hull = cv2.convexHull(contour)
    if len(hull) < 3: 
        return hull.reshape(-1, 2)
    
    pts = hull.reshape(-1, 2)
    pts_wrap = np.vstack([pts, pts[:2]])
    
    smooth_pts = []
    for i in range(len(pts)):
        p_prev = pts_wrap[i]
        p_curr = pts_wrap[i+1]
        p_next = pts_wrap[i+2]
        new_p = 0.25*p_prev + 0.5*p_curr + 0.25*p_next
        smooth_pts.append(new_p)
        
    return np.array(smooth_pts)

def draw_tight_contour(ax, contour, is_left):
    smooth_poly = get_smooth_contour(contour)
    if smooth_poly is None:
        return
    
    color = '#B0C4DE' if is_left else '#F08080' 
    edge_color = '#4070a0' if is_left else '#a04040'
    
    ax.fill(smooth_poly[:, 0], smooth_poly[:, 1], color=color, alpha=0.4, label=f"{'Left' if is_left else 'Right'} Foot Area")
    ax.plot(smooth_poly[:, 0], smooth_poly[:, 1], color=edge_color, linewidth=2, alpha=0.8)

def _save_plot_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# --- Sit to Stand Plots ---

def plot_force_profile(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    ax.plot(s.time_s, s.force, 'w-', label='Total Force')
    
    rise_start = features.get("Rise Start Time (s)")
    rise_end = features.get("Rise End Time (s)")
    
    if rise_start is not None and rise_end is not None and not (isinstance(rise_start, float) and np.isnan(rise_start)):
        ax.axvline(x=rise_start, color='g', linestyle='--', label='Start')
        if rise_end is not None:
            ax.axvline(x=rise_end, color='r', linestyle='--', label='Stable')
            if rise_end > rise_start:
                ax.axvspan(rise_start, rise_end, color='green', alpha=0.1)
                
    ax.set_title("Total Force", color='white')
    ax.set_ylabel("Force (sum)", color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_ap_displacement(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    force = s.force
    active_mask = force > 10.0
    
    if np.any(active_mask):
        active_time = s.time_s[active_mask]
        active_cop_y = s.cop_y[active_mask]
        cop_y_centered = active_cop_y - np.mean(active_cop_y[:10])
        ax.plot(active_time, cop_y_centered, 'c-', label='AP Displacement')
        
        rise_start = features.get("Rise Start Time (s)")
        rise_end = features.get("Rise End Time (s)")
        if rise_start is not None and rise_end is not None and rise_end > rise_start:
             ax.axvspan(rise_start, rise_end, color='green', alpha=0.1, label='Rise Phase')
             
    ax.set_title("Anterior-Posterior Shift", color='white')
    ax.set_ylabel("CoP Y (Forward ->)", color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_stability_map(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    force = s.force
    active_mask = force > 10.0
    
    if np.any(active_mask):
        frames_active = s.frames[active_mask]
        force_active = s.force[active_mask]
        max_force = np.max(force_active) if len(force_active) > 0 else 1.0
        
        avg_frame = np.mean(frames_active, axis=0)
        original_h, original_w = avg_frame.shape
        
        if np.max(avg_frame) > 0:
            norm_avg = (avg_frame / np.max(avg_frame) * 255).astype(np.uint8)
        else:
            norm_avg = avg_frame.astype(np.uint8)
            
        _, thresh_avg = cv2.threshold(norm_avg, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        valid_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    valid_blobs.append({'contour': cnt, 'centroid': (cx, cy), 'area': area})
        
        mask_L = np.zeros_like(avg_frame, dtype=bool)
        mask_R = np.zeros_like(avg_frame, dtype=bool)
        contour_L = None
        contour_R = None
        
        if len(valid_blobs) >= 2:
            foot1 = valid_blobs[0]
            foot2 = valid_blobs[1]
            if foot1['centroid'][0] < foot2['centroid'][0]:
                blob_L, blob_R = foot1, foot2
            else:
                blob_L, blob_R = foot2, foot1
                
            contour_L = blob_L['contour']
            contour_R = blob_R['contour']
            
            m_L = np.zeros((original_h, original_w), dtype=np.uint8)
            m_R = np.zeros((original_h, original_w), dtype=np.uint8)
            cv2.drawContours(m_L, [contour_L], -1, 1, -1)
            cv2.drawContours(m_R, [contour_R], -1, 1, -1)
            mask_L = m_L.astype(bool)
            mask_R = m_R.astype(bool)
        else:
            midline = original_w // 2
            mask_L[:, :midline] = True
            mask_R[:, midline:] = True
        
        def get_centroids_masked(frames_stack, mask):
            masked_frames = frames_stack * mask[None, :, :]
            weights = np.sum(masked_frames, axis=(1, 2))
            valid = weights > 1.0 
            weights[~valid] = 1.0 
            rows, c = frames_stack.shape[1], frames_stack.shape[2]
            r_idx, c_idx = np.indices((rows, c))
            cy = np.sum(masked_frames * r_idx[None, :, :], axis=(1, 2)) / weights
            cx = np.sum(masked_frames * c_idx[None, :, :], axis=(1, 2)) / weights
            return cx, cy, valid

        lx, ly, l_valid = get_centroids_masked(frames_active, mask_L)
        rx, ry, r_valid = get_centroids_masked(frames_active, mask_R)

        if contour_L is not None:
            draw_tight_contour(ax, contour_L, True)
        if contour_R is not None:
            draw_tight_contour(ax, contour_R, False)
        
        is_standing = force_active > (0.8 * max_force)
        
        ax.scatter([], [], c='blue', label='Sit Phase')
        ax.scatter([], [], c='red', label='Stand Phase')
        
        l_sit_mask = l_valid & (~is_standing)
        if np.any(l_sit_mask):
            ax.scatter(lx[l_sit_mask], ly[l_sit_mask], c='blue', s=5, alpha=0.4)
        
        l_stand_mask = l_valid & is_standing
        if np.any(l_stand_mask):
            ax.scatter(lx[l_stand_mask], ly[l_stand_mask], c='red', s=8, alpha=0.6)
            avg_lx = np.mean(lx[l_stand_mask])
            avg_ly = np.mean(ly[l_stand_mask])
            ax.plot(avg_lx, avg_ly, 'w+', markersize=10, markeredgewidth=2)
            
        r_sit_mask = r_valid & (~is_standing)
        if np.any(r_sit_mask):
            ax.scatter(rx[r_sit_mask], ry[r_sit_mask], c='blue', s=5, alpha=0.4)
            
        r_stand_mask = r_valid & is_standing
        if np.any(r_stand_mask):
            ax.scatter(rx[r_stand_mask], ry[r_stand_mask], c='red', s=8, alpha=0.6)
            avg_rx = np.mean(rx[r_stand_mask])
            avg_ry = np.mean(ry[r_stand_mask])
            ax.plot(avg_rx, avg_ry, 'w+', markersize=10, markeredgewidth=2)

    ax.set_title("CoP Stability (Sit -> Stand)", color='white')
    ax.axis('equal')
    ax.set_axis_off()
    ax.legend(loc='best', fontsize='x-small', facecolor='#333333', labelcolor='white')
    
    return _save_plot_to_b64(fig)

# --- Rise to Toes Plots ---

def plot_rise_area(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    # Features may contain area data if we passed it back, but signals has raw area
    ax.plot(s.time_s, s.area, 'b-', label='Contact Area')
    
    baseline = features.get("Baseline Area (pixels)", 0)
    thresh = features.get("Area Threshold", 0)
    
    if baseline > 0:
        ax.axhline(y=baseline, color='w', linestyle='--', label='Flat Baseline')
        ax.axhline(y=thresh, color='r', linestyle=':', label='Toes Threshold')
        
    ax.set_title("Contact Area (Heel Rise)", color='white')
    ax.set_ylabel("Area (pixels)", color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_rise_ap(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    force = s.force
    active_mask = force > 10.0
    
    if np.any(active_mask):
        active_time = s.time_s[active_mask]
        active_cop_y = s.cop_y[active_mask]
        # Center
        cop_y_centered = active_cop_y - np.mean(active_cop_y[:10])
        
        ax.plot(active_time, cop_y_centered, 'purple', label='Forward Shift')
        
    ax.set_title("Anterior-Posterior Balance", color='white')
    ax.set_ylabel("CoP Y Displacement", color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_rise_stability_map(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    force = s.force
    active_mask = force > 10.0
    
    if np.any(active_mask):
        frames_active = s.frames[active_mask]
        
        avg_frame = np.mean(frames_active, axis=0)
        original_h, original_w = avg_frame.shape
        
        if np.max(avg_frame) > 0:
            norm_avg = (avg_frame / np.max(avg_frame) * 255).astype(np.uint8)
        else:
            norm_avg = avg_frame.astype(np.uint8)
            
        _, thresh_avg = cv2.threshold(norm_avg, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        valid_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    valid_blobs.append({'contour': cnt, 'centroid': (cx, cy), 'area': area})
        
        mask_L = np.zeros_like(avg_frame, dtype=bool)
        mask_R = np.zeros_like(avg_frame, dtype=bool)
        contour_L = None
        contour_R = None
        
        if len(valid_blobs) >= 2:
            foot1 = valid_blobs[0]
            foot2 = valid_blobs[1]
            if foot1['centroid'][0] < foot2['centroid'][0]:
                blob_L, blob_R = foot1, foot2
            else:
                blob_L, blob_R = foot2, foot1
            contour_L = blob_L['contour']
            contour_R = blob_R['contour']
            m_L = np.zeros((original_h, original_w), dtype=np.uint8)
            m_R = np.zeros((original_h, original_w), dtype=np.uint8)
            cv2.drawContours(m_L, [contour_L], -1, 1, -1)
            cv2.drawContours(m_R, [contour_R], -1, 1, -1)
            mask_L = m_L.astype(bool)
            mask_R = m_R.astype(bool)
        else:
            midline = original_w // 2
            mask_L[:, :midline] = True
            mask_R[:, midline:] = True
            
        def get_centroids_masked(frames_stack, mask):
            masked_frames = frames_stack * mask[None, :, :]
            weights = np.sum(masked_frames, axis=(1, 2))
            valid = weights > 1.0 
            weights[~valid] = 1.0 
            rows, c = frames_stack.shape[1], frames_stack.shape[2]
            r_idx, c_idx = np.indices((rows, c))
            cy = np.sum(masked_frames * r_idx[None, :, :], axis=(1, 2)) / weights
            cx = np.sum(masked_frames * c_idx[None, :, :], axis=(1, 2)) / weights
            return cx, cy, valid

        lx, ly, l_valid = get_centroids_masked(frames_active, mask_L)
        rx, ry, r_valid = get_centroids_masked(frames_active, mask_R)

        if contour_L is not None:
            draw_tight_contour(ax, contour_L, True)
        if contour_R is not None:
            draw_tight_contour(ax, contour_R, False)
            
        # --- Rise Logic for Coloring ---
        # Use Area Threshold from features to determine "On Toes" (Red) vs "Flat" (Blue)
        thresh_area = features.get("Area Threshold", 0)
        active_area = s.area[active_mask]
        
        # If area < threshold, it's "On Toes" (Red). Else "Flat" (Blue).
        is_on_toes = active_area < thresh_area
        
        ax.scatter([], [], c='blue', label='Flat Phase')
        ax.scatter([], [], c='red', label='Toes Phase')
        
        l_flat = l_valid & (~is_on_toes)
        if np.any(l_flat):
            ax.scatter(lx[l_flat], ly[l_flat], c='blue', s=5, alpha=0.4)
            
        l_toes = l_valid & is_on_toes
        if np.any(l_toes):
            ax.scatter(lx[l_toes], ly[l_toes], c='red', s=8, alpha=0.6)
            ax.plot(np.mean(lx[l_toes]), np.mean(ly[l_toes]), 'w+', markersize=10, markeredgewidth=2)
            
        r_flat = r_valid & (~is_on_toes)
        if np.any(r_flat):
            ax.scatter(rx[r_flat], ry[r_flat], c='blue', s=5, alpha=0.4)
            
        r_toes = r_valid & is_on_toes
        if np.any(r_toes):
            ax.scatter(rx[r_toes], ry[r_toes], c='red', s=8, alpha=0.6)
            ax.plot(np.mean(rx[r_toes]), np.mean(ry[r_toes]), 'w+', markersize=10, markeredgewidth=2)

    ax.set_title("CoP Stability (Center of Pressure)", color='white')
    ax.axis('equal')
    ax.set_axis_off()
    ax.legend(loc='best', fontsize='x-small', facecolor='#333333', labelcolor='white')
    
    return _save_plot_to_b64(fig)

# --- Stance Plots ---

def plot_stance_cop_sway(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    valid_mask = np.isfinite(s.cop_x) & np.isfinite(s.cop_y)
    if np.any(valid_mask):
        time_valid = s.time_s[valid_mask]
        cop_x_valid = s.cop_x[valid_mask]
        cop_y_valid = s.cop_y[valid_mask]
        
        # De-mean to show sway around the center
        cop_x_demeaned = cop_x_valid - np.mean(cop_x_valid)
        cop_y_demeaned = cop_y_valid - np.mean(cop_y_valid)

        ax.plot(time_valid, cop_x_demeaned, 'r-', alpha=0.8, label='M/L Sway (X)')
        ax.plot(time_valid, cop_y_demeaned, 'b-', alpha=0.8, label='A/P Sway (Y)')
    
    ax.set_title("CoP Sway Over Time", color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Sway (sensor units)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_stance_stabilogram(s, features):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    valid_mask = np.isfinite(s.cop_x) & np.isfinite(s.cop_y)
    if np.any(valid_mask):
        cop_x_valid = s.cop_x[valid_mask]
        cop_y_valid = s.cop_y[valid_mask]
        
        # De-mean to center
        cop_x_demeaned = cop_x_valid - np.mean(cop_x_valid)
        cop_y_demeaned = cop_y_valid - np.mean(cop_y_valid)
        
        # Plot the path
        ax.plot(cop_x_demeaned, cop_y_demeaned, 'w-', linewidth=0.8, alpha=0.6, label='CoP Path')
        ax.scatter(cop_x_demeaned, cop_y_demeaned, c=s.time_s[valid_mask], cmap='viridis', s=2, alpha=0.8)
        
        # Add 95% Confidence Ellipse if possible, or just axes
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

    ax.set_title("CoP Stabilogram (X vs Y)", color='white')
    ax.set_xlabel("Medial-Lateral (X)", color='white')
    ax.set_ylabel("Anterior-Posterior (Y)", color='white')
    ax.tick_params(colors='white')
    ax.axis('equal') 
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_stance_stability_map(s, features):
    """
    CoP Stability Map for Stance.
    Shows footprints and individual CoP trajectories for Left and Right feet.
    """
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    force = s.force
    active_mask = force > 10.0
    
    if np.any(active_mask):
        frames_active = s.frames[active_mask]
        
        avg_frame = np.mean(frames_active, axis=0)
        original_h, original_w = avg_frame.shape
        
        # Normalize for contour detection
        if np.max(avg_frame) > 0:
            norm_avg = (avg_frame / np.max(avg_frame) * 255).astype(np.uint8)
        else:
            norm_avg = avg_frame.astype(np.uint8)
            
        _, thresh_avg = cv2.threshold(norm_avg, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        valid_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    valid_blobs.append({'contour': cnt, 'centroid': (cx, cy), 'area': area})
        
        mask_L = np.zeros_like(avg_frame, dtype=bool)
        mask_R = np.zeros_like(avg_frame, dtype=bool)
        contour_L = None
        contour_R = None
        
        # Determine Left vs Right foot
        # If foot1.x < foot2.x -> foot1 is Right (Patient's perspective facing mat?) or just swapped.
        if len(valid_blobs) >= 2:
            foot1 = valid_blobs[0]
            foot2 = valid_blobs[1]
            if foot1['centroid'][0] < foot2['centroid'][0]:
                blob_R, blob_L = foot1, foot2 # SWAPPED as requested
            else:
                blob_R, blob_L = foot2, foot1 # SWAPPED as requested
                
            contour_L = blob_L['contour']
            contour_R = blob_R['contour']
            
            m_L = np.zeros((original_h, original_w), dtype=np.uint8)
            m_R = np.zeros((original_h, original_w), dtype=np.uint8)
            cv2.drawContours(m_L, [contour_L], -1, 1, -1)
            cv2.drawContours(m_R, [contour_R], -1, 1, -1)
            mask_L = m_L.astype(bool)
            mask_R = m_R.astype(bool)
        else:
            midline = original_w // 2
            # Invert logic here too if needed, but midline split is safer to keep standard unless user specifies
            # Assuming standard view: Left on Left, Right on Right of image.
            # If user says swapped, maybe they mean the specific foot detection.
            # Let's stick to the blob logic swap for now.
            mask_L[:, :midline] = True
            mask_R[:, midline:] = True
            if len(valid_blobs) == 1:
                blob = valid_blobs[0]
                if blob['centroid'][0] < midline:
                    contour_L = blob['contour']
                else:
                    contour_R = blob['contour']

        # Draw contours
        if contour_L is not None:
            draw_tight_contour(ax, contour_L, True)
        if contour_R is not None:
            draw_tight_contour(ax, contour_R, False)
            
        # Calculate CoP per foot
        def get_centroids_masked(frames_stack, mask):
            masked_frames = frames_stack * mask[None, :, :]
            weights = np.sum(masked_frames, axis=(1, 2))
            valid = weights > 1.0 
            weights[~valid] = 1.0 
            rows, c = frames_stack.shape[1], frames_stack.shape[2]
            r_idx, c_idx = np.indices((rows, c))
            cy = np.sum(masked_frames * r_idx[None, :, :], axis=(1, 2)) / weights
            cx = np.sum(masked_frames * c_idx[None, :, :], axis=(1, 2)) / weights
            return cx, cy, valid

        lx, ly, l_valid = get_centroids_masked(frames_active, mask_L)
        rx, ry, r_valid = get_centroids_masked(frames_active, mask_R)
        
        # Plot Left Foot CoP (Blue)
        if np.any(l_valid):
            ax.scatter(lx[l_valid], ly[l_valid], c='blue', s=3, alpha=0.3, label='Left CoP')
            ax.plot(np.mean(lx[l_valid]), np.mean(ly[l_valid]), 'w+', markersize=10, markeredgewidth=2) 
            
        # Plot Right Foot CoP (Cyan)
        if np.any(r_valid):
            ax.scatter(rx[r_valid], ry[r_valid], c='cyan', s=3, alpha=0.3, label='Right CoP')
            ax.plot(np.mean(rx[r_valid]), np.mean(ry[r_valid]), 'w+', markersize=10, markeredgewidth=2)

    ax.set_title("CoP Stability Map (Stance)", color='white')
    ax.axis('equal')
    ax.set_axis_off()
    ax.legend(loc='best', fontsize='x-small', facecolor='#333333', labelcolor='white')
    
    return _save_plot_to_b64(fig)


def generate_video(signals, fps=10):
    """
    Generates a smooth GIF animation of the pressure mat frames using PIL.
    Uses explicit savefig approach for maximum robustness across backends.
    Uses MASKING (filtering) instead of slicing to ensure continuous visibility of feet.
    """
    force = signals.force
    
    # Use mask to select only valid frames where feet are present
    # Threshold of 10.0 is standard for pressure mats (noise floor usually ~2-5)
    active_mask = force > 10.0
    
    frames = signals.frames[active_mask]
    times = signals.time_s[active_mask]

    if len(frames) < 2:
         # If filtering removed everything (unlikely), fallback to raw
         frames = signals.frames
         times = signals.time_s
         active_mask = np.ones_like(force, dtype=bool)

    if len(frames) == 0:
        return None

    # Downsample for GIF size (target ~50-80 frames)
    total_frames = len(frames)
    if total_frames > 60:
        step = int(np.ceil(total_frames / 50))
    else:
        step = 1
    
    frames_disp = frames[::step]
    times_disp = times[::step]
    
    # Robust scaling (same as Sit to Stand)
    if np.max(frames_disp) > 0:
        vmax = np.percentile(frames_disp, 99.5)
        if vmax < 1.0: vmax = 1.0
    else:
        vmax = 1.0
    
    # Setup plot for frame generation
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='black')
    ax.set_facecolor('black')
    ax.set_position([0, 0, 1, 1]) 
    ax.axis('off')
    
    # Use 'hot' colormap and nearest interpolation (crisp pixels) to match Sit to Stand
    im_plot = ax.imshow(frames_disp[0], cmap='hot', vmin=0, vmax=vmax, interpolation='nearest', aspect='auto')
    txt = ax.text(0.5, 0.95, f"Time: {times_disp[0]:.2f}s", 
                  size=10, ha="center", transform=ax.transAxes, color='white')
    
    pil_frames = []
    
    try:
        for i in range(len(frames_disp)):
            im_plot.set_data(frames_disp[i])
            txt.set_text(f"Time: {times_disp[i]:.2f}s")
            
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', facecolor='black')
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                pil_frames.append(img)
            
        if pil_frames:
            duration = int(1000 / fps)
            
            with io.BytesIO() as output:
                pil_frames[0].save(
                    output, 
                    format="GIF", 
                    save_all=True, 
                    append_images=pil_frames[1:], 
                    duration=duration, 
                    loop=0
                )
                b64_video = base64.b64encode(output.getvalue()).decode('utf-8')
                return b64_video
                
    except Exception as e:
        print(f"Error generating video with PIL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)
        
    return None

def _get_single_leg_mask(signals):
    """
    Helper to create a boolean mask identifying the single-leg stance phase
    using visual BLOB COUNTING.
    
    Strategy:
    1. Threshold each frame.
    2. Count disconnected blobs > min_size (approx 10 pixels).
    3. If count == 1, it's potentially single leg.
    4. Refine by checking area outliers within the single-blob set.
    """
    force = signals.force
    area = signals.area
    frames = signals.frames
    
    # 1. Initial Mask (On the Mat)
    active_mask = force > 10.0
    
    if not np.any(active_mask):
        return np.zeros_like(force, dtype=bool)
        
    # Initialize blob count mask
    n_frames = len(frames)
    single_blob_mask = np.zeros(n_frames, dtype=bool)
    
    # Process only active frames to save time
    active_indices = np.where(active_mask)[0]
    
    # Downsample for speed if too many frames? 
    # But we need a mask for ALL frames. 
    # Simple loop is fast enough for typical 30s recordings at ~50Hz (1500 frames).
    
    valid_areas_single_blob = []
    
    for i in active_indices:
        frame = frames[i]
        # Normalize frame for thresholding
        if np.max(frame) > 0:
            norm_frame = (frame / np.max(frame) * 255).astype(np.uint8)
            _, thresh = cv2.threshold(norm_frame, 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count significant blobs
            n_blobs = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 10: # Min foot size in pixels
                    n_blobs += 1
            
            if n_blobs == 1:
                single_blob_mask[i] = True
                valid_areas_single_blob.append(area[i])
    
    # 2. Refine using Area Outliers (Two feet touching)
    # If they touch, they form 1 blob but area is large.
    if len(valid_areas_single_blob) > 0:
        median_area = np.median(valid_areas_single_blob)
        # Threshold: 1.6x median (generous for sway, strict enough for two feet)
        size_mask = area < (1.6 * median_area)
        
        final_mask = single_blob_mask & size_mask
    else:
        final_mask = single_blob_mask # Fallback
        
    if not np.any(final_mask):
        # Fallback: If blob counting failed (e.g. very low res), revert to simple area.
        # But visually, blobs are distinct.
        return active_mask 
        
    return final_mask

def plot_one_leg_sway(s, features):
    """Filtered Sway Over Time for One Leg Stance"""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    mask = _get_single_leg_mask(s)
    
    if np.any(mask):
        time_valid = s.time_s[mask]
        cop_x_valid = s.cop_x[mask]
        cop_y_valid = s.cop_y[mask]
        
        cop_x_demeaned = cop_x_valid - np.mean(cop_x_valid)
        cop_y_demeaned = cop_y_valid - np.mean(cop_y_valid)

        ax.plot(time_valid, cop_x_demeaned, 'r-', alpha=0.8, label='M/L Sway (X)')
        ax.plot(time_valid, cop_y_demeaned, 'b-', alpha=0.8, label='A/P Sway (Y)')
    
    ax.set_title("CoP Sway Over Time (Single Leg Phase)", color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Sway (sensor units)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_one_leg_stabilogram(s, features):
    """Filtered Stabilogram for One Leg Stance"""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    mask = _get_single_leg_mask(s)
    
    if np.any(mask):
        cop_x_valid = s.cop_x[mask]
        cop_y_valid = s.cop_y[mask]
        time_valid = s.time_s[mask]
        
        cop_x_demeaned = cop_x_valid - np.mean(cop_x_valid)
        cop_y_demeaned = cop_y_valid - np.mean(cop_y_valid)
        
        ax.plot(cop_x_demeaned, cop_y_demeaned, 'w-', linewidth=0.8, alpha=0.6, label='CoP Path')
        ax.scatter(cop_x_demeaned, cop_y_demeaned, c=time_valid, cmap='viridis', s=2, alpha=0.8)
        
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

    ax.set_title("CoP Stabilogram (Single Leg Phase)", color='white')
    ax.set_xlabel("Medial-Lateral (X)", color='white')
    ax.set_ylabel("Anterior-Posterior (Y)", color='white')
    ax.tick_params(colors='white')
    ax.axis('equal') 
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_one_leg_stability_map(s, features, stance_leg):
    """
    CoP Stability Map for One Leg Stance.
    Isolates the single-leg stance phase using Area and Spatial filtering.
    Shows only the stance foot CoP during the stable(ish) phase.
    """
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_facecolor('none')
    
    mask = _get_single_leg_mask(s)
    
    if np.any(mask):
        frames_to_use = s.frames[mask]
        cop_x_valid = s.cop_x[mask]
        cop_y_valid = s.cop_y[mask]
        
        # Draw Contour from FILTERED frames (Clean Footprint)
        if len(frames_to_use) > 0:
            avg_frame = np.mean(frames_to_use, axis=0)
            
            if np.max(avg_frame) > 0:
                norm_avg = (avg_frame / np.max(avg_frame) * 255).astype(np.uint8)
            else:
                norm_avg = avg_frame.astype(np.uint8)
                
            _, thresh_avg = cv2.threshold(norm_avg, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                is_left = (stance_leg.lower() == "left")
                draw_tight_contour(ax, largest_cnt, is_left)

        # Plot CoP Points
        if len(cop_x_valid) > 0:
            color = 'blue' if stance_leg.lower() == "left" else 'cyan'
            ax.scatter(cop_x_valid, cop_y_valid, c=color, s=3, alpha=0.3, label=f'{stance_leg} CoP')
            ax.plot(np.mean(cop_x_valid), np.mean(cop_y_valid), 'w+', markersize=10, markeredgewidth=2, label='Mean')

    ax.set_title(f"CoP Stability Map ({stance_leg} Leg)\n(Single Leg Phase Only)", color='white', fontsize=10)
    ax.axis('equal')
    ax.set_axis_off()
    ax.legend(loc='best', fontsize='x-small', facecolor='#333333', labelcolor='white')
    
    return _save_plot_to_b64(fig)


# --- FGA Gait Analysis Plots ---

def plot_fga_stride_analysis(signals, features):
    """Plot stride length over time for gait analysis."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')
    ax.set_facecolor('none')
    
    gait_cycles = signals.gait_cycles
    
    if gait_cycles and len(gait_cycles) > 0:
        stride_lengths = []
        cycle_times = []
        
        # Get the first timepoint to calculate relative times
        first_timepoint = None
        if len(signals.time_s) > 0 and hasattr(signals, 'df') and 'timepoint' in signals.df.columns:
            first_timepoint = signals.df['timepoint'].iloc[0]
            if isinstance(first_timepoint, str):
                first_timepoint = pd.to_datetime(first_timepoint)
        
        for cycle in gait_cycles:
            if 'average_horizontal_distance' in cycle and cycle['average_horizontal_distance'] is not None:
                stride_lengths.append(cycle['average_horizontal_distance'])
                
                # Get the actual time for this gait cycle from heel_strike timestamp
                cycle_time = 0.0
                if 'right_leg' in cycle and 'heel_strike' in cycle['right_leg']:
                    heel_strike = cycle['right_leg']['heel_strike']
                    if first_timepoint is not None and heel_strike is not None:
                        if isinstance(heel_strike, pd.Timestamp):
                            cycle_time = (heel_strike - first_timepoint).total_seconds()
                        elif hasattr(heel_strike, '__sub__'):
                            time_diff = heel_strike - first_timepoint
                            cycle_time = time_diff.total_seconds() if hasattr(time_diff, 'total_seconds') else float(time_diff)
                elif 'left_leg' in cycle and 'heel_strike' in cycle['left_leg']:
                    heel_strike = cycle['left_leg']['heel_strike']
                    if first_timepoint is not None and heel_strike is not None:
                        if isinstance(heel_strike, pd.Timestamp):
                            cycle_time = (heel_strike - first_timepoint).total_seconds()
                        elif hasattr(heel_strike, '__sub__'):
                            time_diff = heel_strike - first_timepoint
                            cycle_time = time_diff.total_seconds() if hasattr(time_diff, 'total_seconds') else float(time_diff)
                
                cycle_times.append(cycle_time)
        
        if stride_lengths and len(cycle_times) == len(stride_lengths):
            ax.plot(cycle_times, stride_lengths, 'o-', color='cyan', label='Stride Length', linewidth=2, markersize=6)
            ax.axhline(y=np.mean(stride_lengths), color='yellow', linestyle='--', label=f'Mean: {np.mean(stride_lengths):.1f} cm', alpha=0.7)
    
    ax.set_title("Stride Length Over Time", color='white', fontsize=12)
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Stride Length (cm)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_fga_cadence_analysis(signals, features):
    """Plot cadence over time for gait analysis."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')
    ax.set_facecolor('none')
    
    gait_cycles = signals.gait_cycles
    
    if gait_cycles and len(gait_cycles) > 0:
        cadences = []
        cycle_times = []
        
        # Get the first timepoint to calculate relative times
        first_timepoint = None
        if len(signals.time_s) > 0 and hasattr(signals, 'df') and 'timepoint' in signals.df.columns:
            first_timepoint = signals.df['timepoint'].iloc[0]
            if isinstance(first_timepoint, str):
                first_timepoint = pd.to_datetime(first_timepoint)
        
        for cycle in gait_cycles:
            if 'cadence' in cycle and cycle['cadence'] is not None:
                cadences.append(cycle['cadence'])
                
                # Get the actual time for this gait cycle from heel_strike timestamp
                cycle_time = 0.0
                if 'right_leg' in cycle and 'heel_strike' in cycle['right_leg']:
                    heel_strike = cycle['right_leg']['heel_strike']
                    if first_timepoint is not None and heel_strike is not None:
                        if isinstance(heel_strike, pd.Timestamp):
                            cycle_time = (heel_strike - first_timepoint).total_seconds()
                        elif hasattr(heel_strike, '__sub__'):
                            time_diff = heel_strike - first_timepoint
                            cycle_time = time_diff.total_seconds() if hasattr(time_diff, 'total_seconds') else float(time_diff)
                elif 'left_leg' in cycle and 'heel_strike' in cycle['left_leg']:
                    heel_strike = cycle['left_leg']['heel_strike']
                    if first_timepoint is not None and heel_strike is not None:
                        if isinstance(heel_strike, pd.Timestamp):
                            cycle_time = (heel_strike - first_timepoint).total_seconds()
                        elif hasattr(heel_strike, '__sub__'):
                            time_diff = heel_strike - first_timepoint
                            cycle_time = time_diff.total_seconds() if hasattr(time_diff, 'total_seconds') else float(time_diff)
                
                cycle_times.append(cycle_time)
        
        if cadences and len(cycle_times) == len(cadences):
            ax.plot(cycle_times, cadences, 'o-', color='magenta', label='Cadence', linewidth=2, markersize=6)
            ax.axhline(y=np.mean(cadences), color='yellow', linestyle='--', label=f'Mean: {np.mean(cadences):.1f} steps/min', alpha=0.7)
    
    ax.set_title("Cadence Over Time", color='white', fontsize=12)
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Cadence (steps/min)", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_fga_cop_trajectories(signals, features):
    """Plot CoP trajectories for left and right foot (NOT butterfly plot)."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
    ax.set_facecolor('none')
    
    cop_traces = signals.cop_traces
    
    # Extract left and right CoP traces
    # Note: cop_traces uses 'x' and 'y' keys, not 'cop_x' and 'cop_y'
    left_cop_x = []
    left_cop_y = []
    right_cop_x = []
    right_cop_y = []
    
    for trace in cop_traces:
        if 'foot' in trace and 'x' in trace and 'y' in trace:
            foot = trace['foot']
            cop_x = trace['x']
            cop_y = trace['y']
            
            # FIX: Swap left and right (user says they're swapped)
            if foot == 'left':
                # This is actually right foot
                if isinstance(cop_x, list):
                    right_cop_x.extend(cop_x)
                else:
                    right_cop_x.append(cop_x)
                if isinstance(cop_y, list):
                    right_cop_y.extend(cop_y)
                else:
                    right_cop_y.append(cop_y)
            elif foot == 'right':
                # This is actually left foot
                if isinstance(cop_x, list):
                    left_cop_x.extend(cop_x)
                else:
                    left_cop_x.append(cop_x)
                if isinstance(cop_y, list):
                    left_cop_y.extend(cop_y)
                else:
                    left_cop_y.append(cop_y)
    
    # Plot left foot CoP (blue) - now correctly swapped
    if left_cop_x and left_cop_y:
        left_cop_x_arr = np.array(left_cop_x)
        left_cop_y_arr = np.array(left_cop_y)
        # Center around mean
        left_cop_x_centered = left_cop_x_arr - np.mean(left_cop_x_arr)
        left_cop_y_centered = left_cop_y_arr - np.mean(left_cop_y_arr)
        ax.plot(left_cop_x_centered, left_cop_y_centered, 'b-', alpha=0.6, linewidth=1.5, label='Left Foot CoP')
        ax.scatter(left_cop_x_centered, left_cop_y_centered, c='blue', s=2, alpha=0.4)
        ax.plot(np.mean(left_cop_x_centered), np.mean(left_cop_y_centered), 'b+', markersize=12, markeredgewidth=2, label='Left Mean')
    
    # Plot right foot CoP (red) - now correctly swapped
    if right_cop_x and right_cop_y:
        right_cop_x_arr = np.array(right_cop_x)
        right_cop_y_arr = np.array(right_cop_y)
        # Center around mean
        right_cop_x_centered = right_cop_x_arr - np.mean(right_cop_x_arr)
        right_cop_y_centered = right_cop_y_arr - np.mean(right_cop_y_arr)
        ax.plot(right_cop_x_centered, right_cop_y_centered, 'r-', alpha=0.6, linewidth=1.5, label='Right Foot CoP')
        ax.scatter(right_cop_x_centered, right_cop_y_centered, c='red', s=2, alpha=0.4)
        ax.plot(np.mean(right_cop_x_centered), np.mean(right_cop_y_centered), 'r+', markersize=12, markeredgewidth=2, label='Right Mean')
    
    # Add center lines
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_title("CoP Trajectories (Left vs Right Foot)", color='white', fontsize=12)
    ax.set_xlabel("Medial-Lateral (X) - Centered", color='white')
    ax.set_ylabel("Anterior-Posterior (Y) - Centered", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def plot_fga_butterfly_cop(signals, features):
    """
    Plot proper butterfly CoP plot showing continuous CoP trace during gait.
    The butterfly shape shows the double support phase at the intersection.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
    ax.set_facecolor('none')
    
    # Calculate overall CoP from all frames (continuous trace)
    frames = signals.frames
    time_s = signals.time_s
    
    if len(frames) == 0:
        return _save_plot_to_b64(fig)
    
    # Filter to active frames only
    active_frames = []
    active_times = []
    for i, frame in enumerate(frames):
        if np.sum(frame) > 10.0:
            active_frames.append(frame)
            active_times.append(time_s[i])
    
    if len(active_frames) == 0:
        return _save_plot_to_b64(fig)
    
    # Calculate CoP for each active frame
    cop_x_all = []
    cop_y_all = []
    
    for frame in active_frames:
        rows, cols = frame.shape
        row_indices = np.arange(rows).reshape(rows, 1)
        col_indices = np.arange(cols).reshape(1, cols)
        
        total_force = np.sum(frame)
        if total_force > 0:
            cop_y = np.sum(frame * row_indices) / total_force
            cop_x = np.sum(frame * col_indices) / total_force
            cop_x_all.append(cop_x)
            cop_y_all.append(cop_y)
    
    if len(cop_x_all) == 0:
        return _save_plot_to_b64(fig)
    
    # Convert to arrays
    cop_x_arr = np.array(cop_x_all)
    cop_y_arr = np.array(cop_y_all)
    
    # Center around mean for visualization
    cop_x_centered = cop_x_arr - np.mean(cop_x_arr)
    cop_y_centered = cop_y_arr - np.mean(cop_y_arr)
    
    # Plot continuous CoP trace (butterfly shape)
    # Color by time to show progression
    time_colors = active_times[:len(cop_x_centered)] if len(active_times) >= len(cop_x_centered) else active_times + [active_times[-1]] * (len(cop_x_centered) - len(active_times))
    scatter = ax.scatter(cop_x_centered, cop_y_centered, c=time_colors, 
                        cmap='viridis', s=3, alpha=0.6, label='CoP Trace')
    
    # Plot the continuous line
    ax.plot(cop_x_centered, cop_y_centered, 'w-', linewidth=1.0, alpha=0.4, label='CoP Path')
    
    # Mark the center (double support phase intersection)
    ax.plot(0, 0, 'ro', markersize=8, markeredgecolor='yellow', markeredgewidth=2, 
            label='Center (Double Support)', zorder=10)
    
    # Add center lines
    ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.3)
    
    # Add colorbar for time progression
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)', color='white')
    cbar.ax.tick_params(colors='white')
    
    ax.set_title("Butterfly CoP Plot (Continuous CoP Trace)", color='white', fontsize=12)
    ax.set_xlabel("Medial-Lateral (X) - Centered", color='white')
    ax.set_ylabel("Anterior-Posterior (Y) - Centered", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc="best", fontsize='small', facecolor='#333333', labelcolor='white')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    return _save_plot_to_b64(fig)

def generate_fga_video(signals, fps=10):
    """
    Generate video replay showing gait across 6 mats (treadmill view).
    Uses same 'hot' colormap (black-red-yellow) as MiniBEST exercises.
    """
    frames = signals.frames
    time_s = signals.time_s
    
    if len(frames) == 0:
        return None
    
    # Filter frames to only show those with meaningful pressure (like MiniBEST)
    active_frames = []
    active_times = []
    for i, frame in enumerate(frames):
        if np.sum(frame) > 10.0:  # Threshold for meaningful pressure
            active_frames.append(frame)
            active_times.append(time_s[i])
    
    if len(active_frames) < 2:
        # Fallback to all frames if filtering removed too much
        active_frames = frames
        active_times = time_s
    
    # Downsample for GIF size
    total_frames = len(active_frames)
    if total_frames > 80:
        step = int(np.ceil(total_frames / 60))
    else:
        step = 1
    
    frames_disp = active_frames[::step]
    times_disp = np.array(active_times)[::step]
    
    # Robust scaling (same as MiniBEST)
    if len(frames_disp) > 0 and np.max(frames_disp) > 0:
        vmax = np.percentile([np.max(f) for f in frames_disp], 99.5)
        if vmax < 1.0:
            vmax = 1.0
    else:
        vmax = 1.0
    
    # Setup plot for frame generation - wider for 6 mats
    fig, ax = plt.subplots(figsize=(12, 3), facecolor='black')
    ax.set_facecolor('black')
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    
    # Use 'hot' colormap (black-red-yellow) - same as MiniBEST
    im_plot = ax.imshow(frames_disp[0], cmap='hot', vmin=0, vmax=vmax, interpolation='nearest', aspect='auto')
    txt = ax.text(0.5, 0.95, f"Time: {times_disp[0]:.2f}s", 
                  size=12, ha="center", transform=ax.transAxes, color='white', weight='bold')
    
    pil_frames = []
    
    try:
        for i in range(len(frames_disp)):
            im_plot.set_data(frames_disp[i])
            txt.set_text(f"Time: {times_disp[i]:.2f}s")
            
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', facecolor='black', bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                pil_frames.append(img)
        
        if pil_frames:
            duration = int(1000 / fps)
            
            with io.BytesIO() as output:
                pil_frames[0].save(
                    output,
                    format="GIF",
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration,
                    loop=0
                )
                b64_video = base64.b64encode(output.getvalue()).decode('utf-8')
                return b64_video
                
    except Exception as e:
        print(f"Error generating FGA video: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)
    
    return None

def generate_plot_components(exercise_type, signals, features):
    """
    Returns a dictionary of plot components (base64 strings).
    """
    components = {}
    
    if exercise_type == "sit_to_stand":
        components["force_plot"] = plot_force_profile(signals, features)
        components["ap_plot"] = plot_ap_displacement(signals, features)
        components["stability_plot"] = plot_stability_map(signals, features)
        
    elif exercise_type == "rise_to_toes":
        components["force_plot"] = plot_rise_area(signals, features) # Area vs Time (Top Right)
        components["ap_plot"] = plot_rise_ap(signals, features) # AP Balance (Bottom Left)
        components["stability_plot"] = plot_rise_stability_map(signals, features) # Stability Map (Bottom Right)
        
    elif exercise_type in ["stance_eyes_open", "stance_eyes_closed"]:
        components["force_plot"] = plot_stance_cop_sway(signals, features) # Sway over time
        components["ap_plot"] = plot_stance_stabilogram(signals, features) # Stabilogram (X vs Y)
        components["stability_plot"] = plot_stance_stability_map(signals, features) # CoP Map
        
    elif exercise_type in ["stand_one_leg_left", "stand_one_leg_right"]:
        leg = "Left" if "left" in exercise_type else "Right"
        components["force_plot"] = plot_one_leg_sway(signals, features) # Filtered Sway
        components["ap_plot"] = plot_one_leg_stabilogram(signals, features) # Filtered Stabilogram
        components["stability_plot"] = plot_one_leg_stability_map(signals, features, leg) # Filtered Map

    return components

def generate_fga_plot_components(exercise_num, signals, features):
    """
    Generate FGA-specific plot components for all exercises.
    """
    components = {}
    
    try:
        # Stride analysis plot (for all exercises)
        components["stride_plot"] = plot_fga_stride_analysis(signals, features)
    except Exception as e:
        print(f"Error generating stride plot: {e}")
        components["stride_plot"] = None
    
    try:
        # Cadence analysis plot (for all exercises)
        components["cadence_plot"] = plot_fga_cadence_analysis(signals, features)
    except Exception as e:
        print(f"Error generating cadence plot: {e}")
        components["cadence_plot"] = None
    
    try:
        # CoP Trajectories plot (for all exercises)
        components["cop_trajectories_plot"] = plot_fga_cop_trajectories(signals, features)
    except Exception as e:
        print(f"Error generating CoP trajectories plot: {e}")
        components["cop_trajectories_plot"] = None
    
    # Butterfly CoP plot (for forward walking exercises)
    if exercise_num in [1, 2, 3, 4, 8, 9]:
        try:
            components["butterfly_cop_plot"] = plot_fga_butterfly_cop(signals, features)
        except Exception as e:
            print(f"Error generating butterfly CoP plot: {e}")
            components["butterfly_cop_plot"] = None
    
    return components