import ast
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

@dataclass
class BasicMatSignals:
    """Container for basic signals derived from one mat CSV."""
    time_s: np.ndarray          # shape (T,)
    force: np.ndarray           # total force (sum of all sensors), shape (T,)
    cop_x: np.ndarray           # center of pressure (column index units), shape (T,)
    cop_y: np.ndarray           # center of pressure (row index units), shape (T,)
    dt: float                   # median sampling period (seconds)
    area: np.ndarray            # number of active sensors (contact area proxy), shape (T,)
    frames: np.ndarray          # Full 3D sensor data: (T, rows, cols)


@dataclass
class ExerciseResult:
    participant: str
    exercise: str
    variant: Optional[str]      # direction / side, e.g. BACKWARD, LEFT, RIGHT
    file_path: str
    score: int                  # MiniBEST-style 0–2
    features: Dict[str, float]


# ---------------------------
# Low-level data utilities
# ---------------------------

def _parse_sensors_column(df: pd.DataFrame) -> np.ndarray:
    """
    Parse the 'sensors' column (stringified 2D list) into a 3D numpy array.
    Returns array of shape (T, rows, cols).
    """
    if "sensors" not in df.columns:
        raise ValueError("CSV does not contain 'sensors' column.")

    sensors_list: List[List[List[float]]] = []
    for raw in df["sensors"]:
        sensors_list.append(ast.literal_eval(raw))

    sensors_arr = np.array(sensors_list, dtype=float)
    if sensors_arr.ndim != 3:
        raise ValueError(f"Expected sensors array with 3 dims, got shape {sensors_arr.shape}")
    return sensors_arr


def load_basic_signals_from_df(df: pd.DataFrame) -> BasicMatSignals:
    """Load basic signals from a DataFrame already in memory."""
    # Sort by timepoint to fix disordered rows/negative dt
    if "timepoint" in df.columns:
        df["timepoint"] = pd.to_datetime(df["timepoint"])
        df = df.sort_values("timepoint").reset_index(drop=True)

    # Time in seconds from start of trial
    if "timepoint" not in df.columns:
        raise ValueError("CSV does not contain 'timepoint' column.")
    
    time = df["timepoint"]
    time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()

    sensors = _parse_sensors_column(df)
    
    # Apply noise threshold to sensors - LOWERED TO 0.5 to capture low pressure signals
    sensors[sensors < 0.5] = 0.0
    
    rows = sensors.shape[1]
    cols = sensors.shape[2]

    # Total force per sample
    force = sensors.sum(axis=(1, 2))

    # Contact area proxy
    area_threshold = 1e-3
    area = (sensors > area_threshold).sum(axis=(1, 2))

    # Center of pressure
    row_indices = np.arange(rows).reshape(1, rows, 1)
    col_indices = np.arange(cols).reshape(1, 1, cols)

    eps = 1e-6
    cop_y = (sensors * row_indices).sum(axis=(1, 2)) / (force + eps)
    cop_x = (sensors * col_indices).sum(axis=(1, 2)) / (force + eps)

    cop_y[force < 1e-3] = np.nan
    cop_x[force < 1e-3] = np.nan

    if len(time_s) > 1:
        diffs = np.diff(time_s)
        valid_diffs = diffs[diffs > 0.0001]
        if len(valid_diffs) > 0:
            dt = float(np.median(valid_diffs))
        else:
            dt = 0.01
    else:
        dt = 0.01

    return BasicMatSignals(
        time_s=time_s,
        force=force,
        cop_x=cop_x,
        cop_y=cop_y,
        dt=dt,
        area=area,
        frames=sensors
    )

def load_basic_signals(csv_path: str) -> BasicMatSignals:
    df = pd.read_csv(csv_path)
    return load_basic_signals_from_df(df)

def _sway_metrics(cop_x: np.ndarray, cop_y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(cop_x) & np.isfinite(cop_y)
    if not mask.any():
        return {"cop_path_length": float("nan"), "cop_rms": float("nan")}

    x = cop_x[mask]
    y = cop_y[mask]
    dx = np.diff(x)
    dy = np.diff(y)
    path = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    x0 = x - x.mean()
    y0 = y - y.mean()
    rms = float(np.sqrt(np.mean(x0 * x0 + y0 * y0)))
    return {"cop_path_length": path, "cop_rms": rms}

def _duration(signals: BasicMatSignals) -> float:
    return float(signals.time_s[-1] - signals.time_s[0]) if len(signals.time_s) > 1 else 0.0

def _stance_balance_metrics(signals: BasicMatSignals) -> Dict[str, float]:
    force = signals.force.astype(float)
    area = signals.area.astype(float)
    dt = max(signals.dt, 1e-4)

    active_force_thresh = 10.0
    active_mask = force > active_force_thresh

    valid_pressure_mask = active_mask & (area > 0)
    if np.any(valid_pressure_mask):
        pressure_per_sensor = force[valid_pressure_mask] / area[valid_pressure_mask]
        avg_pressure_per_sensor = float(np.mean(pressure_per_sensor))
    else:
        avg_pressure_per_sensor = float("nan")

    if np.any(valid_pressure_mask):
        baseline_area = float(np.median(area[valid_pressure_mask]))
    else:
        baseline_area = float("nan")

    if np.any(active_mask):
        median_force = float(np.median(force[active_mask]))
    else:
        median_force = 0.0

    if np.isfinite(baseline_area) and baseline_area > 0 and median_force > 0:
        low_area_thresh = 0.7 * baseline_area
        low_force_thresh = 0.5 * median_force
        loss_mask = active_mask & ((area < low_area_thresh) | (force < low_force_thresh))
    else:
        loss_mask = np.zeros_like(force, dtype=bool)

    min_event_duration = 0.2
    min_event_frames = max(int(round(min_event_duration / dt)), 1)

    loss_count = 0
    run_length = 0
    for flag in loss_mask:
        if flag:
            run_length += 1
        else:
            if run_length >= min_event_frames:
                loss_count += 1
            run_length = 0
    if run_length >= min_event_frames:
        loss_count += 1

    return {
        "Number of Balance Losses": float(loss_count),
        "Average Pressure / Active Sensor": avg_pressure_per_sensor,
        "Baseline Area (pixels)": baseline_area,
    }

# CV Helpers
def _get_blobs(frame: np.ndarray) -> List[Dict]:
    if np.max(frame) > 0:
        norm_frame = (frame / np.max(frame) * 255).astype(np.uint8)
    else:
        return []
    
    _, thresh = cv2.threshold(norm_frame, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3: 
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        blobs.append({
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area,
            "contour": cnt
        })
        
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs

def _track_step_count_cv(signals: BasicMatSignals) -> Tuple[int, float]:
    frames = signals.frames
    times = signals.time_s
    
    step_count = 0
    first_step_time = float("nan")
    
    n_baseline = min(5, len(frames))
    baseline_blobs = []
    for i in range(n_baseline):
        blobs = _get_blobs(frames[i])
        baseline_blobs.extend(blobs[:2])
        
    if not baseline_blobs:
        return 0, float("nan")
        
    prev_centroids = [b['centroid'] for b in _get_blobs(frames[0])[:2]]
    in_step = False
    move_thresh = 3.0 
    
    for i in range(1, len(frames)):
        curr_blobs = _get_blobs(frames[i])
        curr_centroids = [b['centroid'] for b in curr_blobs[:2]]
        
        if not curr_centroids:
            continue
            
        moved = False
        for c in curr_centroids:
            if not prev_centroids:
                dist = 999
            else:
                dist = min(np.sqrt((c[0]-p[0])**2 + (c[1]-p[1])**2) for p in prev_centroids)
            
            if dist > move_thresh:
                moved = True
                break
                
        if moved:
            if not in_step:
                step_count += 1
                in_step = True
                if np.isnan(first_step_time):
                    first_step_time = float(times[i])
        else:
            pass
            
        prev_centroids = curr_centroids
        
        if not moved:
            in_step = False
            
    return step_count, first_step_time

def _fill_small_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return mask
    filled = mask.copy()
    last_true = None
    for idx, flag in enumerate(mask):
        if flag:
            if last_true is not None:
                gap = idx - last_true - 1
                if 0 < gap <= max_gap_frames:
                    filled[last_true + 1 : idx] = True
            last_true = idx
    return filled

def _longest_true_run(mask: np.ndarray, dt: float) -> float:
    longest = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
        else:
            if current > longest:
                longest = current
            current = 0
    if current > longest:
        longest = current
    return longest * dt

def _compute_single_leg_mask(signals: BasicMatSignals) -> np.ndarray:
    force = signals.force
    area = signals.area.astype(float)
    frames = signals.frames
    active_mask = force > 10.0

    if not np.any(active_mask):
        return np.zeros_like(force, dtype=bool)

    active_area = area[active_mask]
    if len(active_area) < 10:
        return active_mask.copy()

    filter_size = max(3, int(0.15 / max(signals.dt, 1e-3)))
    if filter_size % 2 == 0:
        filter_size += 1
    smoothed_area = median_filter(active_area, size=filter_size)

    c1 = np.percentile(smoothed_area, 20)
    c2 = np.percentile(smoothed_area, 80)
    if np.isclose(c1, c2):
        c2 = c1 + 1.0

    for _ in range(15):
        dist1 = np.abs(smoothed_area - c1)
        dist2 = np.abs(smoothed_area - c2)
        cluster1 = dist1 <= dist2
        cluster2 = ~cluster1
        if not cluster1.any() or not cluster2.any():
            break
        new_c1 = smoothed_area[cluster1].mean()
        new_c2 = smoothed_area[cluster2].mean()
        if np.isclose(new_c1, c1) and np.isclose(new_c2, c2):
            break
        c1, c2 = new_c1, new_c2

    if c1 > c2:
        c1, c2 = c2, c1

    spread = abs(c2 - c1)
    if spread < 0.1 * max(c2, 1.0):
        threshold = np.percentile(smoothed_area, 40)
    else:
        threshold = (c1 + c2) / 2.0

    area_mask = np.zeros_like(force, dtype=bool)
    area_mask[active_mask] = smoothed_area <= threshold

    blob_mask = np.zeros_like(force, dtype=bool)
    indices = np.where(active_mask)[0]
    for idx in indices:
        frame = frames[idx]
        if np.max(frame) <= 0:
            continue
        norm_frame = (frame / np.max(frame) * 255).astype(np.uint8)
        _, thresh = cv2.threshold(norm_frame, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        if len(significant) == 1:
            blob_mask[idx] = True

    return area_mask | blob_mask

# Exercise Processors

def process_compensatory_stepping(signals: BasicMatSignals, direction: str, participant: str) -> ExerciseResult:
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    step_count, first_step_time = _track_step_count_cv(signals)
    
    dt = max(signals.dt, 1e-4)
    dx = np.diff(signals.cop_x)
    dy = np.diff(signals.cop_y)
    speed = np.sqrt(dx * dx + dy * dy) / dt
    window = max(int(round(0.5 / dt)), 1)
    
    if len(speed) >= window:
        kernel = np.ones(window) / window
        speed_smoothed = np.convolve(speed, kernel, mode="same")
        speed_thresh = 0.5
        stable_idx = np.where(speed_smoothed < speed_thresh)[0]
        stabilization_time = float(signals.time_s[stable_idx[0]]) if stable_idx.size > 0 else duration
    else:
        stabilization_time = duration

    if step_count == 0:
        score = 0
    else:
        if direction == "FORWARD":
            if step_count <= 2: score = 2
            else: score = 1
        else:
            if step_count == 1: score = 2
            else: score = 1

    features = {
        "Duration (s)": duration,
        "Stabilization Time (s)": stabilization_time,
        "Reaction Time (s)": first_step_time,
        "Number of Steps (CV)": float(step_count),
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }

    return ExerciseResult(participant, "Compensatory stepping correction", direction, "", score, features)

def process_rise_to_toes(signals: BasicMatSignals, participant: str) -> ExerciseResult:
    duration = _duration(signals)
    area = signals.area.astype(float)
    force = signals.force
    time_s = signals.time_s
    
    nonzero_force = force[force > 10]
    if len(nonzero_force) == 0:
         return ExerciseResult(participant, "Rise to toes", None, "", 0, {})
         
    median_force = np.median(nonzero_force)
    active_thresh = median_force * 0.1
    active_mask = force > active_thresh
    
    if not np.any(active_mask):
        return ExerciseResult(participant, "Rise to toes", None, "", 0, {})

    valid_area = area[active_mask]
    valid_time = time_s[active_mask]

    baseline_area = float(np.percentile(valid_area, 95))
    thresh_ratio = 0.80
    area_thresh = baseline_area * thresh_ratio
    
    dt = max(signals.dt, 0.001)
    window_size_sec = 0.2
    window_size = max(3, int(window_size_sec / dt))
    smoothed_area = median_filter(valid_area, size=window_size)
    
    is_on_toes = smoothed_area < area_thresh
    
    max_contiguous_duration = 0.0
    current_run_duration = 0.0
    is_running = False
    max_run_start_idx = 0
    max_run_end_idx = 0
    current_run_start_idx = 0

    for i in range(len(valid_time)):
        if i > 0:
            step_dt = valid_time[i] - valid_time[i-1]
            if step_dt > 0.5:
                is_running = False
                current_run_duration = 0.0
        
        if is_on_toes[i]:
            if not is_running:
                is_running = True
                current_run_start_idx = i
                current_run_duration = 0.0
            elif i > 0:
                current_run_duration += (valid_time[i] - valid_time[i-1])
            
            if current_run_duration > max_contiguous_duration:
                max_contiguous_duration = current_run_duration
                max_run_start_idx = current_run_start_idx
                max_run_end_idx = i
        else:
            is_running = False
            current_run_duration = 0.0
            
    stability_score = 0.0
    if max_contiguous_duration > 0:
        active_indices = np.where(active_mask)[0]
        if max_run_end_idx > max_run_start_idx:
            run_indices = active_indices[max_run_start_idx : max_run_end_idx+1]
            run_cop_x = signals.cop_x[run_indices]
            run_cop_y = signals.cop_y[run_indices]
            dx = np.diff(run_cop_x)
            dy = np.diff(run_cop_y)
            speed = np.sqrt(dx**2 + dy**2) / dt
            valid_speed = speed[speed < 500]
            if len(valid_speed) > 0:
                stability_score = float(np.mean(valid_speed))
    
    is_on_toes_full = np.zeros_like(active_mask, dtype=bool)
    is_on_toes_full[active_mask] = is_on_toes
    flat_mask = active_mask & (~is_on_toes_full)
    flat_global_indices = np.where(flat_mask)[0]
    
    baseline_sway_speed = 0.0
    if len(flat_global_indices) > 10:
        dx = np.diff(signals.cop_x)
        dy = np.diff(signals.cop_y)
        full_speed = np.sqrt(dx**2 + dy**2) / dt
        full_speed = np.concatenate(([0], full_speed))
        flat_speeds = full_speed[flat_global_indices]
        flat_speeds = flat_speeds[(flat_speeds < 500) & np.isfinite(flat_speeds)]
        if len(flat_speeds) > 0:
            baseline_sway_speed = float(np.mean(flat_speeds))
            
    base_thresh = 40.0
    if baseline_sway_speed > 0:
        instability_threshold = max(base_thresh, baseline_sway_speed * 2.5)
    else:
        instability_threshold = 50.0
    
    if max_contiguous_duration < 3.0:
        score = 0
    else:
        if stability_score > instability_threshold:
            score = 1
        else:
            score = 2

    features = {
        "Trial Duration (s)": duration,
        "Max Contiguous On-Toes Duration (s)": max_contiguous_duration,
        "Stability (Mean Sway Speed)": stability_score,
        "Baseline Sway Speed (Flat)": baseline_sway_speed,
        "Adaptive Instability Threshold": instability_threshold,
        "Baseline Area (pixels)": baseline_area,
        "Area Threshold": area_thresh,
    }

    return ExerciseResult(participant, "Rise to toes", None, "", score, features)

def process_sit_to_stand(signals: BasicMatSignals, participant: str, used_hands: bool = False, multiple_attempts: bool = False) -> ExerciseResult:
    time_s = signals.time_s
    f = signals.force
    max_f = np.max(f)
    
    # Use a moving average to smooth the force
    window = max(int(0.5 / signals.dt), 3)
    f_smooth = np.convolve(f, np.ones(window)/window, mode='same')
    
    # 1. Robust Event Detection via Peak Force
    # Anchoring to the peak allows us to find the rise even if the file is cut short or starts late.
    peak_idx = np.argmax(f_smooth)
    peak_force = f_smooth[peak_idx]
    
    # Find "Sitting" Baseline (Low force BEFORE peak)
    # Look back up to 3 seconds from peak
    search_back_samples = int(3.0 / signals.dt)
    start_search_idx = max(0, peak_idx - search_back_samples)
    pre_peak_segment = f_smooth[start_search_idx:peak_idx]
    
    if len(pre_peak_segment) > 0:
        sit_baseline = np.min(pre_peak_segment)
    else:
        sit_baseline = f_smooth[0] if len(f_smooth) > 0 else 0
        
    # Find "Standing" Plateau (Stable force AFTER peak, usually body weight)
    # Standing force is typically slightly less than the peak (momentum overshoot)
    # Look forward up to 3 seconds from peak
    search_fwd_samples = int(3.0 / signals.dt)
    end_search_idx = min(len(f_smooth), peak_idx + search_fwd_samples)
    post_peak_segment = f_smooth[peak_idx:end_search_idx]
    
    if len(post_peak_segment) > 0:
        # Median of the post-peak segment gives a good estimate of stable standing weight
        stand_baseline = np.median(post_peak_segment)
    else:
        stand_baseline = peak_force
        
    # Calculate Dynamic Range
    force_range = stand_baseline - sit_baseline
    
    # Safety check: if range is too small, maybe they were already standing?
    if force_range < 50: 
        sit_baseline = 0 # Assume sit starts at 0
        force_range = stand_baseline
    
    # 2. Define Thresholds
    # Start: Force exceeds Sitting + 15% of range
    start_thresh = sit_baseline + 0.15 * force_range
    # End: Force reaches Sitting + 90% of range (near standing weight)
    end_thresh = sit_baseline + 0.90 * force_range
    
    # 3. Search for Events relative to Peak
    # Find Start: Scan BACKWARDS from peak
    start_idx = 0
    for i in range(peak_idx, -1, -1):
        if f_smooth[i] < start_thresh:
            start_idx = i
            break
            
    # Find End: Scan FORWARDS from start_idx
    # We want the point where it firmly establishes itself in the high zone
    end_idx = peak_idx # Default to peak if we can't find a stable plateau crossing
    for i in range(start_idx, len(f_smooth)):
        if f_smooth[i] > end_thresh:
            end_idx = i
            # Continue scanning to ensure it's not just a spike? 
            # For simple rise time, first crossing is standard.
            break
            
    start_time = float(time_s[start_idx])
    end_time = float(time_s[end_idx])
    
    rise_duration = end_time - start_time
    if rise_duration < 0: rise_duration = 0.0
        
    if multiple_attempts:
        score = 0
    else:
        if max_f < 20: 
            score = 0
        elif used_hands:
            score = 1
        else:
            score = 2
        
    features = {
        "Rise Time (s)": rise_duration,
        "Rise Start Time (s)": start_time,
        "Rise End Time (s)": end_time,
        "Max Force": float(max_f),
        "Used Hands": used_hands,
        "Multiple Attempts": multiple_attempts
    }

    return ExerciseResult(participant, "Sit to Stand", None, "", score, features)

def process_stand_on_one_leg(signals: BasicMatSignals, participant: str, stance_leg: str) -> ExerciseResult:
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)

    mask = _compute_single_leg_mask(signals)
    single_leg_duration = 0.0
    if np.any(mask):
        gap_frames = int(1.0 / max(signals.dt, 1e-3))
        filled_mask = _fill_small_gaps(mask, gap_frames)
        single_leg_duration = float(_longest_true_run(filled_mask, signals.dt))

    if single_leg_duration >= 20.0: score = 2
    elif single_leg_duration >= 10.0: score = 1
    else: score = 0

    features = {
        "Duration (s)": duration,
        "Single Leg Stance Duration (s)": round(single_leg_duration, 2),
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }

    return ExerciseResult(participant, "Stand on one leg", stance_leg, "", score, features)

def process_stance_eyes_open(signals: BasicMatSignals, participant: str) -> ExerciseResult:
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    stance_metrics = _stance_balance_metrics(signals)
    
    step_count, first_step_time = _track_step_count_cv(signals)
    
    if np.isnan(first_step_time):
        effective_stance_duration = duration
    else:
        effective_stance_duration = first_step_time

    if effective_stance_duration >= 30.0: score = 2
    elif effective_stance_duration >= 2.0: score = 1
    else: score = 0

    features = {
        "Trial Duration (s)": duration,
        "Effective Stance Duration (s)": effective_stance_duration,
        "First Step Time (s)": first_step_time,
        "Number of Steps Detected": float(step_count),
        "Number of Balance Losses": stance_metrics.get("Number of Balance Losses", float("nan")),
        "Average Pressure / Active Sensor": stance_metrics.get("Average Pressure / Active Sensor", float("nan")),
        "Baseline Area (pixels)": stance_metrics.get("Baseline Area (pixels)", float("nan")),
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }

    return ExerciseResult(participant, "Stance feet together (eyes open, firm)", None, "", score, features)

def process_stance_eyes_closed(signals: BasicMatSignals, participant: str) -> ExerciseResult:
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    stance_metrics = _stance_balance_metrics(signals)

    step_count, first_step_time = _track_step_count_cv(signals)

    if np.isnan(first_step_time):
        effective_stance_duration = duration
    else:
        effective_stance_duration = first_step_time

    if effective_stance_duration >= 30.0: score = 2
    elif effective_stance_duration >= 2.0: score = 1
    else: score = 0

    features = {
        "Trial Duration (s)": duration,
        "Effective Stance Duration (s)": effective_stance_duration,
        "First Step Time (s)": first_step_time,
        "Number of Steps Detected": float(step_count),
        "Number of Balance Losses": stance_metrics.get("Number of Balance Losses", float("nan")),
        "Average Pressure / Active Sensor": stance_metrics.get("Average Pressure / Active Sensor", float("nan")),
        "Baseline Area (pixels)": stance_metrics.get("Baseline Area (pixels)", float("nan")),
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }

    return ExerciseResult(participant, "Stance feet together (eyes closed, foam)", None, "", score, features)


# MiniBEST Gait Exercises (10-14) - Use FGA-style analysis with MiniBEST scoring (0-2)
def process_change_gait_speed(csv_path: str, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 10: Change in Gait Speed
    
    (2) Normal: Significantly changes walking speed without imbalance.
    (1) Moderate: Unable to change walking speed or signs of imbalance.
    (0) Severe: Unable to achieve significant change in walking speed AND signs of imbalance.
    """
    from . import fga
    
    signals = fga.load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return ExerciseResult(participant, "Change in Gait Speed", None, csv_path, 0, {})
    
    metrics = fga._calculate_common_metrics(signals, structured_steps)
    
    # Analyze speed changes by looking at cadence variation
    cadences = []
    for cycle in gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    speed_variation = np.std(cadences) if len(cadences) > 1 else 0
    significant_speed_change = speed_variation > 10  # At least 10 steps/min variation
    has_imbalance = metrics["max_deviation_cm"] > 25.4  # Significant deviation indicates imbalance
    
    # MiniBEST scoring (0-2)
    if significant_speed_change and not has_imbalance:
        score = 2
    elif significant_speed_change or not has_imbalance:
        score = 1
    else:
        score = 0
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "speed_variation_steps_per_min": float(speed_variation),
        "significant_speed_change": bool(significant_speed_change),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
    }
    
    return ExerciseResult(participant, "Change in Gait Speed", None, csv_path, score, features)


def process_walk_head_turns_horizontal(csv_path: str, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 11: Walk with Head Turns - Horizontal
    
    (2) Normal: Performs head turns with no change in gait speed and good balance.
    (1) Moderate: Performs head turns with reduction in gait speed.
    (0) Severe: Performs head turns with imbalance.
    """
    from . import fga
    
    signals = fga.load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return ExerciseResult(participant, "Walk with Head Turns - Horizontal", None, csv_path, 0, {})
    
    metrics = fga._calculate_common_metrics(signals, structured_steps)
    
    # Analyze cadence variation (head turns may cause speed changes)
    cadences = []
    for cycle in gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    cadence_variation = np.std(cadences) if len(cadences) > 1 else 0
    speed_reduction = cadence_variation > 15  # High variation indicates speed reduction
    has_imbalance = metrics["max_deviation_cm"] > 25.4
    
    # MiniBEST scoring
    if not speed_reduction and not has_imbalance:
        score = 2
    elif speed_reduction and not has_imbalance:
        score = 1
    else:
        score = 0
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "cadence_variation_steps_per_min": float(cadence_variation),
        "speed_reduction": bool(speed_reduction),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
    }
    
    return ExerciseResult(participant, "Walk with Head Turns - Horizontal", None, csv_path, score, features)


def process_walk_pivot_turns(csv_path: str, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 12: Walk with Pivot Turns
    
    (2) Normal: Turns with feet close FAST (< 3 steps) with good balance.
    (1) Moderate: Turns with feet close SLOW (>4 steps) with good balance.
    (0) Severe: Cannot turn with feet close at any speed without imbalance.
    """
    from . import fga
    
    signals = fga.load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return ExerciseResult(participant, "Walk with Pivot Turns", None, csv_path, 0, {})
    
    metrics = fga._calculate_common_metrics(signals, structured_steps)
    number_of_steps = metrics["number_of_steps"]
    turn_time = metrics["actual_walking_time"]
    has_imbalance = metrics["max_deviation_cm"] > 25.4
    
    # Estimate steps for turn (assuming turn is part of the exercise)
    # If turn_time < 3s and steps < 4, it's fast
    fast_turn = turn_time < 3.0 and number_of_steps < 4
    slow_turn = turn_time >= 3.0 and number_of_steps >= 4
    
    # MiniBEST scoring
    if fast_turn and not has_imbalance:
        score = 2
    elif slow_turn and not has_imbalance:
        score = 1
    else:
        score = 0
    
    features = {
        "turn_time_s": float(turn_time),
        "number_of_steps": int(number_of_steps),
        "total_exercise_time_s": float(metrics["total_time"]),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
        "fast_turn": bool(fast_turn),
    }
    
    return ExerciseResult(participant, "Walk with Pivot Turns", None, csv_path, score, features)


def process_step_over_obstacles(csv_path: str, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 13: Step Over Obstacles
    
    (2) Normal: Able to step over box with minimal change of gait speed and with good balance.
    (1) Moderate: Steps over box but touches box OR displays cautious behavior by slowing gait.
    (0) Severe: Unable to step over box OR steps around box.
    """
    from . import fga
    
    signals = fga.load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return ExerciseResult(participant, "Step Over Obstacles", None, csv_path, 0, {})
    
    metrics = fga._calculate_common_metrics(signals, structured_steps)
    
    # Analyze cadence variation (obstacle may cause speed changes)
    cadences = []
    for cycle in gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    cadence_variation = np.std(cadences) if len(cadences) > 1 else 0
    speed_maintained = cadence_variation < 10  # Low variation = speed maintained
    has_imbalance = metrics["max_deviation_cm"] > 25.4
    
    # MiniBEST scoring
    if speed_maintained and not has_imbalance:
        score = 2
    elif not speed_maintained or has_imbalance:
        score = 1  # Could be touching box or slowing down
    else:
        score = 0  # Unable to step over
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "cadence_variation_steps_per_min": float(cadence_variation),
        "speed_maintained": bool(speed_maintained),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
    }
    
    return ExerciseResult(participant, "Step Over Obstacles", None, csv_path, score, features)


def process_tug_dual_task(csv_path_tug: str, csv_path_dual: str, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 14: Timed Up & Go with Dual Task
    
    (2) Normal: No noticeable change in sitting, standing or walking while backward counting 
        when compared to TUG without Dual Task.
    (1) Moderate: Dual Task affects either counting OR walking (>10%) when compared to the TUG without Dual Task.
    (0) Severe: Stops counting while walking OR stops walking while counting.
    """
    from . import fga
    
    # Process both TUG alone and TUG with dual task
    signals_tug = fga.load_fga_signals(csv_path_tug)
    signals_dual = fga.load_fga_signals(csv_path_dual)
    
    metrics_tug = fga._calculate_common_metrics(signals_tug, signals_tug.structured_steps)
    metrics_dual = fga._calculate_common_metrics(signals_dual, signals_dual.structured_steps)
    
    tug_time = metrics_tug["actual_walking_time"]
    dual_time = metrics_dual["actual_walking_time"]
    
    # Calculate percentage change
    if tug_time > 0:
        time_change_percent = ((dual_time - tug_time) / tug_time) * 100
    else:
        time_change_percent = 0
    
    # Check if walking was maintained (both completed)
    tug_completed = metrics_tug["estimated_distance_m"] >= 2.5  # ~3 meters
    dual_completed = metrics_dual["estimated_distance_m"] >= 2.5
    
    # MiniBEST scoring
    if tug_completed and dual_completed and abs(time_change_percent) < 10:
        score = 2  # No noticeable change
    elif tug_completed and dual_completed and abs(time_change_percent) >= 10:
        score = 1  # Dual task affects walking (>10%)
    else:
        score = 0  # Stops walking or counting
    
    features = {
        "tug_time_s": float(tug_time),
        "dual_task_time_s": float(dual_time),
        "time_change_percent": float(time_change_percent),
        "tug_completed": bool(tug_completed),
        "dual_task_completed": bool(dual_completed),
        "tug_distance_m": float(metrics_tug["estimated_distance_m"]),
        "dual_task_distance_m": float(metrics_dual["estimated_distance_m"]),
        "tug_steps": int(metrics_tug["number_of_steps"]),
        "dual_task_steps": int(metrics_dual["number_of_steps"]),
    }
    
    return ExerciseResult(participant, "Timed Up & Go with Dual Task", None, csv_path_dual, score, features)
