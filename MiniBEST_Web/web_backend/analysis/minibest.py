import ast
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d

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

def _detect_steps_with_size(signals: BasicMatSignals, direction: str) -> Tuple[int, float, List[float]]:
    """
    Detect steps using multiple methods for reliability.
    Returns: (step_count, first_step_time, step_sizes)
    """
    # Try blob-based detection first
    blob_step_count, blob_first_time, blob_step_indices = _detect_steps_blob(signals)
    
    # Always also try CoP-based detection as backup/validation
    cop_step_count, cop_first_time, cop_step_sizes = _detect_steps_from_cop(signals, direction)
    
    # Use the method that detects more steps (or blob if equal, as it's more reliable)
    if blob_step_count > 0:
        # Use blob detection results, but calculate step sizes from CoP
        step_count = blob_step_count
        first_step_time = blob_first_time
        step_sizes = _calculate_step_sizes_from_indices(signals, blob_step_indices, direction)
    elif cop_step_count > 0:
        # Use CoP detection results
        step_count = cop_step_count
        first_step_time = cop_first_time
        step_sizes = cop_step_sizes
    else:
        # Neither detected steps - check for significant CoP movement as last resort
        step_count, first_step_time, step_sizes = _detect_steps_from_cop_movement(signals, direction)
    
    return step_count, first_step_time, step_sizes


def _detect_steps_blob(signals: BasicMatSignals) -> Tuple[int, float, List[int]]:
    """Detect steps using blob tracking."""
    frames = signals.frames
    times = signals.time_s
    
    step_count = 0
    first_step_time = float("nan")
    step_indices = []
    
    n_baseline = min(5, len(frames))
    baseline_blobs = []
    for i in range(n_baseline):
        blobs = _get_blobs(frames[i])
        baseline_blobs.extend(blobs[:2])
    
    if not baseline_blobs:
        return 0, float("nan"), []
    
    prev_centroids = [b['centroid'] for b in _get_blobs(frames[0])[:2]]
    in_step = False
    move_thresh = 2.5  # Lowered from 3.0 for better sensitivity
    
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
                step_indices.append(i)
                in_step = True
                if np.isnan(first_step_time):
                    first_step_time = float(times[i])
        
        prev_centroids = curr_centroids
        
        if not moved:
            in_step = False
    
    return step_count, first_step_time, step_indices


def _calculate_step_sizes_from_indices(signals: BasicMatSignals, step_indices: List[int], direction: str) -> List[float]:
    """Calculate step sizes from step indices using CoP data."""
    cop_x = signals.cop_x
    cop_y = signals.cop_y
    
    step_sizes = []
    
    # Get baseline CoP position
    baseline_frames = min(int(0.5 / signals.dt), len(cop_x) - 1, 5)
    if baseline_frames < 1:
        baseline_frames = 1
    
    baseline_cop_x = np.nanmean(cop_x[:baseline_frames])
    baseline_cop_y = np.nanmean(cop_y[:baseline_frames])
    
    for step_idx in step_indices:
        window_start = max(0, step_idx - int(0.2 / signals.dt))
        window_end = min(len(cop_x), step_idx + int(0.5 / signals.dt))
        
        step_cop_x = cop_x[window_start:window_end]
        step_cop_y = cop_y[window_start:window_end]
        
        valid_x = step_cop_x[~np.isnan(step_cop_x)]
        valid_y = step_cop_y[~np.isnan(step_cop_y)]
        
        if len(valid_x) > 0 and len(valid_y) > 0:
            if direction == "FORWARD" or direction == "BACKWARD":
                step_size = np.max(np.abs(valid_y - baseline_cop_y))
            else:
                step_size = np.max(np.abs(valid_x - baseline_cop_x))
            step_sizes.append(step_size)
        else:
            step_sizes.append(5.0)  # Default estimate
    
    return step_sizes


def _detect_steps_from_cop(signals: BasicMatSignals, direction: str) -> Tuple[int, float, List[float]]:
    """
    Detect steps from CoP movement using speed-based detection.
    """
    times = signals.time_s
    cop_x = signals.cop_x
    cop_y = signals.cop_y
    
    step_count = 0
    first_step_time = float("nan")
    step_sizes = []
    
    # Get baseline CoP position (first 0.5 seconds)
    baseline_frames = min(int(0.5 / signals.dt), len(cop_x) - 1)
    if baseline_frames < 1:
        baseline_frames = 1
    
    baseline_cop_x = np.nanmean(cop_x[:baseline_frames])
    baseline_cop_y = np.nanmean(cop_y[:baseline_frames])
    
    # Detect significant CoP movements (steps) - use very low threshold
    dt = max(signals.dt, 1e-4)
    dx = np.diff(cop_x)
    dy = np.diff(cop_y)
    speed = np.sqrt(dx * dx + dy * dy) / dt
    
    # Smooth speed with larger window
    window = max(int(round(0.2 / dt)), 1)
    if len(speed) >= window:
        kernel = np.ones(window) / window
        speed_smoothed = np.convolve(speed, kernel, mode="same")
    else:
        speed_smoothed = speed
    
    # Very low threshold for better detection
    step_speed_thresh = 0.5  # Lowered from 1.0
    
    # Find step events
    in_step = False
    step_start_idx = 0
    
    for i in range(1, len(speed_smoothed)):
        if speed_smoothed[i] > step_speed_thresh:
            if not in_step:
                in_step = True
                step_start_idx = i
                if np.isnan(first_step_time):
                    first_step_time = float(times[i])
        else:
            if in_step:
                step_end_idx = i
                step_cop_x = cop_x[step_start_idx:step_end_idx]
                step_cop_y = cop_y[step_start_idx:step_end_idx]
                
                valid_x = step_cop_x[~np.isnan(step_cop_x)]
                valid_y = step_cop_y[~np.isnan(step_cop_y)]
                
                if len(valid_x) > 0 and len(valid_y) > 0:
                    if direction == "FORWARD" or direction == "BACKWARD":
                        step_size = np.max(np.abs(valid_y - baseline_cop_y))
                    else:
                        step_size = np.max(np.abs(valid_x - baseline_cop_x))
                    
                    # Only count if step size is significant (> 1.0 sensor units)
                    if step_size > 1.0:
                        step_sizes.append(step_size)
                        step_count += 1
                
                in_step = False
    
    # Handle case where step continues to end
    if in_step:
        step_end_idx = len(cop_x) - 1
        step_cop_x = cop_x[step_start_idx:step_end_idx]
        step_cop_y = cop_y[step_start_idx:step_end_idx]
        
        valid_x = step_cop_x[~np.isnan(step_cop_x)]
        valid_y = step_cop_y[~np.isnan(step_cop_y)]
        
        if len(valid_x) > 0 and len(valid_y) > 0:
            if direction == "FORWARD" or direction == "BACKWARD":
                step_size = np.max(np.abs(valid_y - baseline_cop_y))
            else:
                step_size = np.max(np.abs(valid_x - baseline_cop_x))
            
            if step_size > 1.0:
                step_sizes.append(step_size)
                step_count += 1
    
    return step_count, first_step_time, step_sizes


def _detect_steps_from_cop_movement(signals: BasicMatSignals, direction: str) -> Tuple[int, float, List[float]]:
    """
    Last resort: Detect steps from significant CoP displacement (for cases with large path length).
    """
    times = signals.time_s
    cop_x = signals.cop_x
    cop_y = signals.cop_y
    
    # Get baseline CoP position
    baseline_frames = min(int(0.5 / signals.dt), len(cop_x) - 1)
    if baseline_frames < 1:
        baseline_frames = 1
    
    baseline_cop_x = np.nanmean(cop_x[:baseline_frames])
    baseline_cop_y = np.nanmean(cop_y[:baseline_frames])
    
    # Calculate maximum displacement from baseline
    valid_mask = ~(np.isnan(cop_x) | np.isnan(cop_y))
    if not np.any(valid_mask):
        return 0, float("nan"), []
    
    valid_cop_x = cop_x[valid_mask]
    valid_cop_y = cop_y[valid_mask]
    valid_times = times[valid_mask]
    
    if direction == "FORWARD" or direction == "BACKWARD":
        displacements = np.abs(valid_cop_y - baseline_cop_y)
    else:
        displacements = np.abs(valid_cop_x - baseline_cop_x)
    
    max_displacement = np.max(displacements) if len(displacements) > 0 else 0
    
    # If there's significant displacement (> 5 sensor units), assume at least 1 step
    if max_displacement > 5.0:
        # Find when the displacement first exceeds threshold
        threshold = max_displacement * 0.3  # 30% of max
        step_indices = np.where(displacements > threshold)[0]
        
        if len(step_indices) > 0:
            first_step_idx = step_indices[0]
            first_step_time = float(valid_times[first_step_idx])
            step_sizes = [max_displacement]
            return 1, first_step_time, step_sizes
    
    return 0, float("nan"), []


def process_compensatory_stepping_forward(signals: BasicMatSignals, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 4: Compensatory Stepping Correction - Forward
    
    (2) Normal: Recovers independently with a single, large step (second realignment step is allowed).
    (1) Moderate: More than one step used to recover equilibrium.
    (0) Severe: No step, OR would fall if not caught, OR falls spontaneously.
    """
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    step_count, first_step_time, step_sizes = _detect_steps_with_size(signals, "FORWARD")
    
    # Calculate stabilization time
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
    
    # Scoring: (2) Single large step (second realignment allowed), (1) More than one step, (0) No step
    if step_count == 0:
        score = 0
    elif step_count == 1:
        # Single step - check if it's large
        large_step_threshold = 3.0  # Lowered threshold for better detection
        if len(step_sizes) > 0 and step_sizes[0] >= large_step_threshold:
            score = 2  # Single large step
        else:
            score = 2  # Still score 2 for single step (even if small, it's a recovery step)
    elif step_count == 2:
        # Two steps - check if first is large (second realignment allowed)
        large_step_threshold = 3.0
        if len(step_sizes) > 0 and step_sizes[0] >= large_step_threshold:
            score = 2  # Large step + small realignment
        else:
            score = 1  # Two small steps
    else:
        score = 1  # More than two steps
    
    features = {
        "Duration (s)": duration,
        "Stabilization Time (s)": stabilization_time,
        "Reaction Time (s)": first_step_time if not np.isnan(first_step_time) else 0.0,
        "Number of Steps": float(step_count),
        "Step Sizes (sensor units)": step_sizes,
        "Largest Step Size": float(max(step_sizes)) if step_sizes else 0.0,
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }
    
    return ExerciseResult(participant, "Compensatory stepping correction - Forward", None, "", score, features)


def process_compensatory_stepping_backward(signals: BasicMatSignals, participant: str) -> ExerciseResult:
    """
    MiniBEST Exercise 5: Compensatory Stepping Correction - Backward
    
    (2) Normal: Recovers independently with a single, large step.
    (1) Moderate: More than one step used to recover equilibrium.
    (0) Severe: No step, OR would fall if not caught, OR falls spontaneously.
    """
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    step_count, first_step_time, step_sizes = _detect_steps_with_size(signals, "BACKWARD")
    
    # Calculate stabilization time
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
    
    # Scoring: (2) Single large step, (1) More than one step, (0) No step
    if step_count == 0:
        score = 0
    elif step_count == 1:
        # Single step - check if it's large
        large_step_threshold = 3.0  # Lowered threshold
        if len(step_sizes) > 0 and step_sizes[0] >= large_step_threshold:
            score = 2  # Single large step
        else:
            score = 2  # Still score 2 for single step (recovery step)
    else:
        score = 1  # More than one step
    
    features = {
        "Duration (s)": duration,
        "Stabilization Time (s)": stabilization_time,
        "Reaction Time (s)": first_step_time if not np.isnan(first_step_time) else 0.0,
        "Number of Steps": float(step_count),
        "Step Sizes (sensor units)": step_sizes,
        "Largest Step Size": float(max(step_sizes)) if step_sizes else 0.0,
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }
    
    return ExerciseResult(participant, "Compensatory stepping correction - Backward", None, "", score, features)


def process_compensatory_stepping_lateral(signals: BasicMatSignals, participant: str, side: str = "LEFT") -> ExerciseResult:
    """
    MiniBEST Exercise 6: Compensatory Stepping Correction - Lateral
    
    (2) Normal: Recovers independently with 1 step (crossover or lateral OK).
    (1) Moderate: Several steps to recover equilibrium.
    (0) Severe: Falls, or cannot step.
    
    Note: Test both left and right, use the side with the lowest score.
    """
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    step_count, first_step_time, step_sizes = _detect_steps_with_size(signals, "LATERAL")
    
    # Calculate stabilization time
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
    
    # Check for fall (large CoP displacement or loss of contact)
    fall_detected = False
    if len(signals.force) > 0:
        # Check if force drops significantly (possible fall)
        max_force = np.max(signals.force)
        min_force = np.min(signals.force)
        if max_force > 0 and min_force / max_force < 0.3:
            # Significant force drop might indicate fall
            fall_detected = True
    
    # Scoring: (2) 1 step, (1) Several steps, (0) Falls/cannot step
    if fall_detected or step_count == 0:
        score = 0
    elif step_count == 1:
        score = 2  # 1 step
    else:
        score = 1  # Several steps
    
    features = {
        "Duration (s)": duration,
        "Stabilization Time (s)": stabilization_time,
        "Reaction Time (s)": first_step_time if not np.isnan(first_step_time) else 0.0,
        "Number of Steps": float(step_count),
        "Step Sizes (sensor units)": step_sizes,
        "Largest Step Size": float(max(step_sizes)) if step_sizes else 0.0,
        "Fall Detected": bool(fall_detected),
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
        "Side": side.upper(),
    }
    
    return ExerciseResult(participant, "Compensatory stepping correction - Lateral", side.upper(), "", score, features)


# Keep old function for backward compatibility
def process_compensatory_stepping(signals: BasicMatSignals, direction: str, participant: str) -> ExerciseResult:
    """Legacy function - routes to specific functions based on direction."""
    if direction.upper() == "FORWARD":
        return process_compensatory_stepping_forward(signals, participant)
    elif direction.upper() == "BACKWARD":
        return process_compensatory_stepping_backward(signals, participant)
    elif direction.upper() in ["LATERAL", "LEFT", "RIGHT"]:
        return process_compensatory_stepping_lateral(signals, participant, direction.upper())
    else:
        # Default to forward
        return process_compensatory_stepping_forward(signals, participant)

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


def _detect_pivot_turn_period(signals, structured_steps):
    """
    Detect the pivot turn period by analyzing CoP movement patterns.
    Returns the start and end times of the pivot turn period, and the number of steps during the turn.
    
    A pivot turn typically shows:
    1. Initial forward walking (CoP X increasing)
    2. Turn initiation (CoP X direction change or reversal)
    3. Rotation period (180-degree turn)
    4. Stop after rotation
    
    Returns:
        tuple: (pivot_steps_count, turn_start_time, turn_end_time, turn_duration)
    """
    if not signals.frames or len(signals.frames) == 0:
        return 0, None, None, 0.0
    
    # Calculate CoP X from all frames
    cop_x_all = []
    frame_times = []
    frame_sums = [np.sum(frame) for frame in signals.frames]
    max_pressure = max(frame_sums) if frame_sums else 0
    pressure_threshold = max(10.0, max_pressure * 0.05)
    
    for i, frame in enumerate(signals.frames):
        if np.sum(frame) > pressure_threshold:
            rows, cols = frame.shape  # rows=48 (width), cols=288 (length)
            col_indices = np.arange(cols).reshape(1, cols)
            
            total_force = np.sum(frame)
            if total_force > 0:
                cop_x = np.sum(frame * col_indices) / total_force
                cop_x_all.append(cop_x)
                frame_times.append(signals.time_s[i])
    
    if len(cop_x_all) < 10:  # Need enough data points
        return 0, None, None, 0.0
    
    cop_x_arr = np.array(cop_x_all)
    frame_times_arr = np.array(frame_times)
    
    # Detect CoP X direction reversal (forward to backward or vice versa)
    # During a pivot turn, the CoP X should reverse direction
    if len(cop_x_arr) > 1:
        cop_x_velocity = np.diff(cop_x_arr)
        
        # Smooth the velocity to reduce noise
        dt = signals.dt if hasattr(signals, 'dt') else (frame_times_arr[-1] - frame_times_arr[0]) / len(frame_times_arr) if len(frame_times_arr) > 1 else 0.01
        window_size = max(3, int(0.2 / max(dt, 0.01)))  # 0.2 second window
        if window_size % 2 == 0:
            window_size += 1
        if len(cop_x_velocity) >= window_size:
            cop_x_velocity_smooth = uniform_filter1d(cop_x_velocity, size=window_size, mode='nearest')
        else:
            cop_x_velocity_smooth = cop_x_velocity
        
        # Find where velocity changes sign significantly (direction reversal)
        velocity_threshold = np.std(cop_x_velocity_smooth) * 0.5 if len(cop_x_velocity_smooth) > 1 else 1.0
        
        # Find the maximum forward position (likely before turn starts)
        max_forward_idx = np.argmax(cop_x_arr)
        
        # Find where velocity becomes negative after max forward (turn starts)
        turn_start_idx = None
        for i in range(max_forward_idx, len(cop_x_velocity_smooth)):
            if cop_x_velocity_smooth[i] < -velocity_threshold:
                turn_start_idx = i
                break
        
        if turn_start_idx is None:
            # Fallback: use middle of the sequence as turn start
            turn_start_idx = len(cop_x_arr) // 3
        
        # Find where movement stabilizes (turn ends)
        # Look for where velocity becomes small and stays small
        turn_end_idx = len(cop_x_arr) - 1
        for i in range(turn_start_idx + 5, len(cop_x_velocity_smooth)):
            # Check if velocity is small for a sustained period
            if i + 10 < len(cop_x_velocity_smooth):
                window_velocity = cop_x_velocity_smooth[i:i+10]
                if np.abs(window_velocity).mean() < velocity_threshold * 0.3:
                    turn_end_idx = min(i + 5, len(cop_x_arr) - 1)  # End of stabilization period
                    break
    else:
        # Fallback: use middle portion as turn period
        turn_start_idx = len(cop_x_arr) // 4
        turn_end_idx = 3 * len(cop_x_arr) // 4
    
    # Convert frame indices to times (in seconds from start)
    if turn_start_idx < len(frame_times_arr) and turn_end_idx < len(frame_times_arr):
        turn_start_time = frame_times_arr[turn_start_idx]
        turn_end_time = frame_times_arr[turn_end_idx]
        turn_duration = turn_end_time - turn_start_time
    else:
        turn_start_time = frame_times_arr[0] if len(frame_times_arr) > 0 else 0.0
        turn_end_time = frame_times_arr[-1] if len(frame_times_arr) > 0 else 0.0
        turn_duration = turn_end_time - turn_start_time
    
    # Count steps during the pivot turn period
    pivot_steps_count = 0
    if structured_steps and turn_start_time is not None and turn_end_time is not None:
        import pandas as pd
        base_time = signals.time_s[0] if len(signals.time_s) > 0 else 0.0
        
        for step in structured_steps:
            step_start = step.get('start_time')
            step_end = step.get('end_time')
            
            if step_start is not None and step_end is not None:
                # Convert step times to seconds from base
                try:
                    # Get base timepoint from first frame if available
                    base_timepoint = None
                    if hasattr(signals, 'df') and signals.df is not None and len(signals.df) > 0:
                        if 'timepoint' in signals.df.columns:
                            base_timepoint = pd.to_datetime(signals.df.iloc[0]['timepoint'])
                    elif len(frame_times_arr) > 0:
                        # Use first frame time as base
                        base_timepoint = pd.Timestamp.fromtimestamp(base_time) if base_time > 0 else None
                    
                    if isinstance(step_start, pd.Timestamp):
                        if base_timepoint is not None:
                            step_start_sec = (step_start - base_timepoint).total_seconds()
                        else:
                            # Fallback: use relative time from step's own timepoint
                            step_start_sec = 0.0
                    else:
                        step_start_sec = float(step_start)
                    
                    if isinstance(step_end, pd.Timestamp):
                        if base_timepoint is not None:
                            step_end_sec = (step_end - base_timepoint).total_seconds()
                        else:
                            step_end_sec = 0.0
                    else:
                        step_end_sec = float(step_end)
                    
                    # Check overlap: step overlaps if it starts before turn ends and ends after turn starts
                    if step_start_sec <= turn_end_time and step_end_sec >= turn_start_time:
                        pivot_steps_count += 1
                except Exception:
                    # If conversion fails, try simpler approach: check if step has frames in turn period
                    step_frames = step.get('frames', [])
                    if step_frames:
                        # Check if any frame time falls within turn period
                        for frame_data in step_frames:
                            frame_timepoint = frame_data.get('timepoint')
                            if frame_timepoint is not None:
                                try:
                                    if isinstance(frame_timepoint, pd.Timestamp):
                                        if base_timepoint is not None:
                                            frame_sec = (frame_timepoint - base_timepoint).total_seconds()
                                        else:
                                            continue
                                    else:
                                        frame_sec = float(frame_timepoint)
                                    
                                    if turn_start_time <= frame_sec <= turn_end_time:
                                        pivot_steps_count += 1
                                        break  # Count step once
                                except:
                                    continue
    
    return pivot_steps_count, turn_start_time, turn_end_time, turn_duration


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
    has_imbalance = metrics["max_deviation_cm"] > 25.4
    
    # Detect pivot turn period and count steps ONLY during the turn
    pivot_steps_count, turn_start_time, turn_end_time, turn_duration = _detect_pivot_turn_period(signals, structured_steps)
    
    # Fallback: if detection failed, use middle portion of steps as pivot turn
    if pivot_steps_count == 0 and structured_steps and len(structured_steps) > 0:
        # Use middle 60% of steps as pivot turn period
        sorted_steps = sorted(structured_steps, key=lambda s: s.get('start_time', pd.Timestamp.min))
        start_idx = len(sorted_steps) // 5  # Start at 20%
        end_idx = 4 * len(sorted_steps) // 5  # End at 80%
        pivot_steps_count = end_idx - start_idx
        if pivot_steps_count == 0:
            pivot_steps_count = len(sorted_steps)  # If still 0, use all steps
    
    # Scoring based on pivot turn steps ONLY
    # (2) Normal: < 3 steps during pivot turn
    # (1) Moderate: >4 steps during pivot turn
    # (0) Severe: Cannot turn without imbalance
    
    if has_imbalance:
        score = 0
    elif pivot_steps_count < 3:
        score = 2  # Fast turn (< 3 steps)
    elif pivot_steps_count >= 4:
        score = 1  # Slow turn (>= 4 steps)
    else:
        # Edge case: exactly 3 steps - could be fast or slow depending on time
        # If turn is fast (< 3s), treat as fast; otherwise slow
        if turn_duration < 3.0:
            score = 2
        else:
            score = 1
    
    features = {
        "pivot_turn_steps": int(pivot_steps_count),  # Steps ONLY during pivot turn
        "total_steps": int(metrics["number_of_steps"]),  # Total steps in entire exercise
        "turn_duration_s": float(turn_duration),
        "total_exercise_time_s": float(metrics["total_time"]),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
        "turn_start_time_s": float(turn_start_time) if turn_start_time is not None else None,
        "turn_end_time_s": float(turn_end_time) if turn_end_time is not None else None,
    }
    
    return ExerciseResult(participant, "Walk with Pivot Turns", None, csv_path, score, features)


def _filter_static_obstacle_steps(structured_steps, signals):
    """
    Filter out steps that are actually the static obstacle in the middle of the treadmill.
    The obstacle is static (doesn't move) and appears at approximately the middle position.
    
    Args:
        structured_steps: List of structured steps
        signals: FGASignals object
    
    Returns:
        Filtered list of steps (obstacle steps removed)
    """
    if not structured_steps or len(structured_steps) == 0:
        return structured_steps
    
    # The obstacle is typically in the middle of the treadmill
    # 6 mats = 288 columns total, middle is around columns 96-192 (mats 2-4)
    total_cols = 288  # 6 mats * 48 columns each
    obstacle_col_start = total_cols // 3  # ~96 (start of mat 2)
    obstacle_col_end = 2 * total_cols // 3  # ~192 (end of mat 4)
    
    # Calculate total exercise time
    total_time = signals.time_s[-1] - signals.time_s[0] if len(signals.time_s) > 1 else 1.0
    
    filtered_steps = []
    
    for step in structured_steps:
        # Get the overbox (bounding box) for this step
        overbox = step.get('metadata', {}).get('overbox')
        if not overbox:
            # If no overbox, calculate it from frames
            frames = step.get('frames', [])
            if not frames:
                continue
            min_col = min(frame.get('bbox', [0, 0, 0, 0])[1] for frame in frames)
            max_col = max(frame.get('bbox', [0, 0, 0, 0])[3] for frame in frames)
        else:
            min_row, min_col, max_row, max_col = overbox
        
        # Calculate step duration
        step_duration = 0
        if step.get('start_time') and step.get('end_time'):
            try:
                import pandas as pd
                if isinstance(step['start_time'], pd.Timestamp) and isinstance(step['end_time'], pd.Timestamp):
                    step_duration = (step['end_time'] - step['start_time']).total_seconds()
                else:
                    step_duration = float(step['end_time']) - float(step['start_time'])
            except:
                step_duration = len(step.get('frames', [])) * 0.1  # Estimate
        
        # Check if step is in obstacle region (middle of treadmill)
        step_center_col = (min_col + max_col) / 2
        is_in_obstacle_region = (obstacle_col_start <= step_center_col <= obstacle_col_end)
        
        # Calculate movement of the step (how much the center moves)
        frames = step.get('frames', [])
        movement = 0
        if len(frames) >= 3:
            centers = []
            for frame in frames:
                bbox = frame.get('bbox')
                if bbox:
                    min_row_f, min_col_f, max_row_f, max_col_f = bbox
                    center_col = (min_col_f + max_col_f) / 2
                    centers.append(center_col)
            
            if len(centers) > 1:
                movement = max(centers) - min(centers)
        
        # Identify obstacle steps
        # The obstacle is: in middle region, static (< 12 columns movement), and long duration
        is_static = movement < 12  # Very little movement (increased threshold to catch more)
        is_long_duration = step_duration > total_time * 0.2  # Present for >20% of exercise (lowered threshold)
        
        # Count how many frames of this step are in the obstacle region
        frames_in_obstacle = 0
        total_frames = len(frames)
        if total_frames > 0:
            for frame in frames:
                bbox = frame.get('bbox')
                if bbox:
                    min_row_f, min_col_f, max_row_f, max_col_f = bbox
                    frame_center_col = (min_col_f + max_col_f) / 2
                    if obstacle_col_start <= frame_center_col <= obstacle_col_end:
                        frames_in_obstacle += 1
            
            # If >60% of frames are in obstacle region, it's likely the obstacle
            obstacle_ratio = frames_in_obstacle / total_frames if total_frames > 0 else 0
        else:
            obstacle_ratio = 0
        
        # Filter out obstacle steps - be very aggressive
        # The obstacle appears as multiple steps in the middle region
        # Remove ALL steps that meet ANY of these criteria:
        # 1. In obstacle region AND static (< 12 columns movement) - regardless of duration
        # 2. In obstacle region AND long duration (>15% of exercise) - regardless of movement
        # 3. Primarily in obstacle region (>30% of frames) - regardless of other criteria
        # 4. In obstacle region AND moderate duration (>10% of exercise) AND static
        is_mostly_in_obstacle = obstacle_ratio > 0.3  # Lowered to catch more
        is_long_enough = step_duration > total_time * 0.15  # Lowered to catch more
        is_moderate_duration = step_duration > total_time * 0.1  # Lowered to catch more
        
        # Remove if step is in obstacle region AND meets any obstacle criteria
        # This should catch all obstacle-related steps, including those that are split into multiple detections
        if is_in_obstacle_region and (is_static or is_long_enough or is_mostly_in_obstacle or (is_moderate_duration and is_static)):
            # This is likely the obstacle - skip it
            continue
        
        # Include all other steps (real walking steps)
        filtered_steps.append(step)
    
    return filtered_steps


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
    
    if not structured_steps or len(structured_steps) == 0:
        return ExerciseResult(participant, "Step Over Obstacles", None, csv_path, 0, {})
    
    # Filter out the static obstacle from steps
    filtered_steps = _filter_static_obstacle_steps(structured_steps, signals)
    
    if len(filtered_steps) == 0:
        return ExerciseResult(participant, "Step Over Obstacles", None, csv_path, 0, {
            "error": "No valid steps detected after filtering obstacle"
        })
    
    # Recalculate metrics using filtered steps (without obstacle)
    metrics = fga._calculate_common_metrics(signals, filtered_steps)
    
    # Filter gait cycles to only include those corresponding to filtered steps
    filtered_gait_cycles = []
    if gait_cycles:
        # Get step heel strikes from filtered steps for matching
        filtered_heel_strikes = set()
        for step in filtered_steps:
            heel_strike = step.get('metadata', {}).get('heel_strike')
            if heel_strike:
                filtered_heel_strikes.add(heel_strike)
        
        for cycle in gait_cycles:
            # Check if cycle's heel strikes match filtered steps
            right_heel = cycle.get('right_leg', {}).get('heel_strike')
            left_heel = cycle.get('left_leg', {}).get('heel_strike')
            
            # Include cycle if both heel strikes are in filtered steps
            if right_heel in filtered_heel_strikes and left_heel in filtered_heel_strikes:
                filtered_gait_cycles.append(cycle)
    
    # Analyze cadence variation (obstacle may cause speed changes)
    cadences = []
    for cycle in filtered_gait_cycles if filtered_gait_cycles else gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    # Calculate cadence before and after obstacle region
    # Obstacle is in middle, so split cadences into before/after
    if len(cadences) >= 4:
        mid_point = len(cadences) // 2
        cadences_before = cadences[:mid_point]
        cadences_after = cadences[mid_point:]
        
        avg_cadence_before = np.mean(cadences_before) if cadences_before else 0
        avg_cadence_after = np.mean(cadences_after) if cadences_after else 0
        
        if avg_cadence_before > 0:
            cadence_change_percent = abs(avg_cadence_after - avg_cadence_before) / avg_cadence_before * 100
        else:
            cadence_change_percent = 0
    else:
        cadence_change_percent = 0
        avg_cadence_before = np.mean(cadences) if cadences else 0
        avg_cadence_after = avg_cadence_before
    
    cadence_variation = np.std(cadences) if len(cadences) > 1 else 0
    
    # Minimal change in gait speed: cadence change < 15% (regardless of variation)
    # This means the patient maintained their speed when stepping over the obstacle
    minimal_speed_change = cadence_change_percent < 15
    # Slowing gait: cadence decreases significantly (> 15%) indicating cautious behavior
    slowing_gait = cadence_change_percent > 15
    
    has_imbalance = metrics["max_deviation_cm"] > 25.4
    
    # Check if patient stepped around obstacle (lateral deviation increases significantly near obstacle)
    # This would show as increased deviation in the middle region
    cop_y_positions = []
    if len(signals.frames) > 0:
        frame_sums = [np.sum(frame) for frame in signals.frames]
        max_pressure = max(frame_sums) if frame_sums else 0
        pressure_threshold = max(10.0, max_pressure * 0.05)
        
        for i, frame in enumerate(signals.frames):
            if np.sum(frame) > pressure_threshold:
                rows, cols = frame.shape
                row_indices = np.arange(rows).reshape(rows, 1)
                total_force = np.sum(frame)
                if total_force > 0:
                    cop_y = np.sum(frame * row_indices) / total_force
                    cop_y_positions.append((signals.time_s[i], cop_y))
    
    stepped_around = False
    if len(cop_y_positions) >= 10:
        times, cop_ys = zip(*cop_y_positions)
        times = np.array(times)
        cop_ys = np.array(cop_ys)
        
        # Find middle time period (where obstacle is)
        mid_time = (times[-1] + times[0]) / 2
        mid_start = mid_time - 1.0  # 1 second before middle
        mid_end = mid_time + 1.0    # 1 second after middle
        
        mid_mask = (times >= mid_start) & (times <= mid_end)
        if np.any(mid_mask):
            mid_deviation = np.std(cop_ys[mid_mask])
            overall_deviation = np.std(cop_ys)
            # If deviation in middle is significantly higher, they might have stepped around
            if mid_deviation > overall_deviation * 1.5 and mid_deviation > 10:
                stepped_around = True
    
    # MiniBEST scoring
    # Score 2: Minimal change in gait speed (< 15% change) AND good balance
    # Score 1: Slowing gait (> 15% decrease) OR imbalance (touching box or cautious behavior)
    # Score 0: Stepped around obstacle OR unable to step over
    
    if stepped_around or len(filtered_steps) < 3:
        # Unable to step over or steps around
        score = 0
    elif minimal_speed_change and not has_imbalance:
        # Minimal change in gait speed (< 15%) and good balance
        # Example: cadence 92 -> 89 is only 3.3% change, which is minimal
        score = 2
    elif has_imbalance:
        # Imbalance indicates touching box or cautious behavior
        score = 1
    elif slowing_gait:
        # Significant slowing (> 15% decrease) indicates cautious behavior
        score = 1
    else:
        # Default case: if cadence change is minimal (< 15%), treat as score 2
        # This handles edge cases where minimal_speed_change might not be set correctly
        if cadence_change_percent < 15 and not has_imbalance:
            score = 2
        else:
            score = 1
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(len(filtered_steps)),  # Real steps (obstacle filtered out)
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "cadence_variation_steps_per_min": float(cadence_variation),
        "cadence_change_percent": float(cadence_change_percent),
        "avg_cadence_before_obstacle": float(avg_cadence_before),
        "avg_cadence_after_obstacle": float(avg_cadence_after),
        "minimal_speed_change": bool(minimal_speed_change),
        "slowing_gait": bool(slowing_gait),
        "stepped_around": bool(stepped_around),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
    }
    
    # Update signals with filtered data for plotting (only update steps, keep original cycles for now)
    signals.structured_steps = filtered_steps
    # Keep original gait cycles but filter out obstacle-related ones for plotting
    if filtered_gait_cycles:
        signals.gait_cycles = filtered_gait_cycles
    else:
        signals.gait_cycles = gait_cycles
    
    # Add flag to indicate signals have been filtered
    features["_use_filtered_steps"] = True
    
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
