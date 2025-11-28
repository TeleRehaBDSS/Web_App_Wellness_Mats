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
        # Each entry looks like: "[[0, 0, ...], [...], ...]"
        sensors_list.append(ast.literal_eval(raw))

    sensors_arr = np.array(sensors_list, dtype=float)
    if sensors_arr.ndim != 3:
        raise ValueError(f"Expected sensors array with 3 dims, got shape {sensors_arr.shape}")
    return sensors_arr


def load_basic_signals(csv_path: str) -> BasicMatSignals:
    """Load one mat CSV and compute basic signals: total force and CoP."""
    df = pd.read_csv(csv_path)

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
    
    # Apply noise threshold to sensors
    # Zero out very small pressures that might be background noise
    sensors[sensors < 2.0] = 0.0
    
    # sensors: (T, rows, cols)
    rows = sensors.shape[1]
    cols = sensors.shape[2]

    # Total force per sample
    force = sensors.sum(axis=(1, 2))

    # Contact area proxy: number of sensors above a small threshold
    area_threshold = 1e-3
    area = (sensors > area_threshold).sum(axis=(1, 2))

    # Center of pressure (index units, not physical units)
    row_indices = np.arange(rows).reshape(1, rows, 1)
    col_indices = np.arange(cols).reshape(1, 1, cols)

    # Avoid division by zero
    eps = 1e-6
    cop_y = (sensors * row_indices).sum(axis=(1, 2)) / (force + eps)
    cop_x = (sensors * col_indices).sum(axis=(1, 2)) / (force + eps)

    # Handle NaN/Inf in CoP caused by zero force frames
    # Set to 0 or interpolate? For now, just keeping them as is, 
    # but we should probably mask them later.
    cop_y[force < 1e-3] = np.nan
    cop_x[force < 1e-3] = np.nan

    # Median sampling interval
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


def _sway_metrics(cop_x: np.ndarray, cop_y: np.ndarray) -> Dict[str, float]:
    """Compute simple sway metrics from CoP trajectory."""
    # Use only finite points
    mask = np.isfinite(cop_x) & np.isfinite(cop_y)
    if not mask.any():
        return {"cop_path_length": float("nan"), "cop_rms": float("nan")}

    x = cop_x[mask]
    y = cop_y[mask]
    # Path length
    dx = np.diff(x)
    dy = np.diff(y)
    path = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    # RMS distance from mean
    x0 = x - x.mean()
    y0 = y - y.mean()
    rms = float(np.sqrt(np.mean(x0 * x0 + y0 * y0)))
    return {"cop_path_length": path, "cop_rms": rms}


def _duration(signals: BasicMatSignals) -> float:
    return float(signals.time_s[-1] - signals.time_s[0]) if len(signals.time_s) > 1 else 0.0


def _stance_balance_metrics(signals: BasicMatSignals) -> Dict[str, float]:
    """
    Compute stance-specific metrics that are shared between
    eyes open / eyes closed variants.

    This helper focuses on *quality* of stance rather than duration:
    - Number of balance-loss episodes
    - Average pressure per active sensor
    - Baseline contact area
    """
    force = signals.force.astype(float)
    area = signals.area.astype(float)
    dt = max(signals.dt, 1e-4)

    # Consider frames with meaningful force as "on the mat"
    active_force_thresh = 10.0
    active_mask = force > active_force_thresh

    # Average pressure per active sensor (force / number of active sensors)
    valid_pressure_mask = active_mask & (area > 0)
    if np.any(valid_pressure_mask):
        pressure_per_sensor = force[valid_pressure_mask] / area[valid_pressure_mask]
        avg_pressure_per_sensor = float(np.mean(pressure_per_sensor))
    else:
        avg_pressure_per_sensor = float("nan")

    # Baseline contact area during stance (use median to be robust to outliers)
    if np.any(valid_pressure_mask):
        baseline_area = float(np.median(area[valid_pressure_mask]))
    else:
        baseline_area = float("nan")

    # Heuristic "loss of balance" detector:
    # Flag frames where area or force drops substantially relative to baseline.
    if np.any(active_mask):
        median_force = float(np.median(force[active_mask]))
    else:
        median_force = 0.0

    # If we do not have a sensible baseline, we will treat all frames as stable.
    if np.isfinite(baseline_area) and baseline_area > 0 and median_force > 0:
        low_area_thresh = 0.7 * baseline_area
        low_force_thresh = 0.5 * median_force
        loss_mask = active_mask & ((area < low_area_thresh) | (force < low_force_thresh))
    else:
        loss_mask = np.zeros_like(force, dtype=bool)

    # Count contiguous episodes of "loss of balance" that last at least a short duration.
    min_event_duration = 0.2  # seconds
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


# ---------------------------
# CV / Image Processing Helpers
# ---------------------------

def _get_blobs(frame: np.ndarray) -> List[Dict]:
    """
    Detect connected components (feet) in a pressure frame using OpenCV.
    Returns list of dicts with keys: 'centroid', 'bbox', 'area'.
    """
    # Normalize frame to 0-255 uint8
    if np.max(frame) > 0:
        norm_frame = (frame / np.max(frame) * 255).astype(np.uint8)
    else:
        return []
    
    # Threshold to binary (low threshold to catch any contact)
    _, thresh = cv2.threshold(norm_frame, 5, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small noise (single pixels or tiny clusters)
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
        
    # Sort by area descending (largest feet first)
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs


def _track_step_count_cv(signals: BasicMatSignals) -> Tuple[int, float]:
    """
    Count steps using blob tracking.
    Returns (step_count, first_step_time).
    """
    frames = signals.frames
    times = signals.time_s
    
    step_count = 0
    first_step_time = float("nan")
    
    # Baseline: Average of first 5 frames (or less if short)
    n_baseline = min(5, len(frames))
    baseline_blobs = []
    for i in range(n_baseline):
        blobs = _get_blobs(frames[i])
        baseline_blobs.extend(blobs[:2])
        
    if not baseline_blobs:
        return 0, float("nan")
        
    # Detection loop
    prev_centroids = [b['centroid'] for b in _get_blobs(frames[0])[:2]]
    
    in_step = False
    
    # Distance threshold (in grid units). Mat is 48x48. 
    move_thresh = 3.0 
    
    for i in range(1, len(frames)):
        curr_blobs = _get_blobs(frames[i])
        curr_centroids = [b['centroid'] for b in curr_blobs[:2]] # Track top 2
        
        if not curr_centroids:
            continue
            
        # Match current centroids to previous nearest neighbors
        moved = False
        
        for c in curr_centroids:
            # Min distance to any previous centroid
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


# ---------------------------
# Scoring helpers
# ---------------------------

def _three_level_score(
    value: float,
    good_threshold: float,
    moderate_threshold: float,
    reverse: bool = False,
) -> int:
    """
    Map a continuous metric to {0,1,2}.
    """
    if np.isnan(value):
        return 0

    if reverse:
        if value <= good_threshold:
            return 2
        if value <= moderate_threshold:
            return 1
        return 0
    else:
        if value >= good_threshold:
            return 2
        if value >= moderate_threshold:
            return 1
        return 0


# ---------------------------
# Exercise-specific analyses
# ---------------------------

def process_compensatory_stepping(csv_path: str, direction: str, participant: str) -> ExerciseResult:
    """
    Compensatory Stepping Correction.
    """
    signals = load_basic_signals(csv_path)
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    
    # Use CV based step counting
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

    # Scoring
    if step_count == 0:
        score = 0
    else:
        if direction == "FORWARD":
            if step_count <= 2:
                score = 2
            else:
                score = 1
        else:
            if step_count == 1:
                score = 2
            else:
                score = 1

    features = {
        "Duration (s)": duration,
        "Stabilization Time (s)": stabilization_time,
        "Reaction Time (s)": first_step_time,
        "Number of Steps (CV)": float(step_count),
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }

    return ExerciseResult(
        participant=participant,
        exercise="Compensatory stepping correction",
        variant=direction,
        file_path=csv_path,
        score=score,
        features=features,
    )


def process_rise_to_toes(csv_path: str, participant: str) -> ExerciseResult:
    """
    Rise to toes.
    Scoring:
    (2) Normal: Stable for 3 s with maximum height.
    (1) Moderate: Heels up, but not full range, OR noticeable instability for 3 s.
    (0) Severe: < 3 s.
    """
    signals = load_basic_signals(csv_path)
    duration = _duration(signals)
    
    area = signals.area.astype(float)
    force = signals.force
    time_s = signals.time_s
    
    # 1. Identify Active Phase (Robust to outliers/noise)
    nonzero_force = force[force > 10] # Minimal noise floor
    if len(nonzero_force) == 0:
         return ExerciseResult(participant, "Rise to toes", None, csv_path, 0, {})
         
    median_force = np.median(nonzero_force)
    active_thresh = median_force * 0.1 # 10% of median valid force
    active_mask = force > active_thresh
    
    if not np.any(active_mask):
        return ExerciseResult(participant, "Rise to toes", None, csv_path, 0, {})

    valid_area = area[active_mask]
    valid_time = time_s[active_mask]

    # 2. Determine Baseline Area (Flat Foot)
    baseline_area = float(np.percentile(valid_area, 95))
    
    # 3. Threshold for "On Toes"
    # Use a stricter threshold to ensure they are really up.
    thresh_ratio = 0.80
    area_thresh = baseline_area * thresh_ratio
    
    # 4. Smooth the Area signal
    dt = max(signals.dt, 0.001)
    window_size_sec = 0.2
    window_size = max(3, int(window_size_sec / dt))
    smoothed_area = median_filter(valid_area, size=window_size)
    
    # 5. Detection of "On Toes" state
    is_on_toes = smoothed_area < area_thresh
    
    # 6. Calculate MAX CONTIGUOUS Duration
    max_contiguous_duration = 0.0
    current_run_duration = 0.0
    max_run_start_idx = 0
    max_run_end_idx = 0
    
    current_run_start_idx = 0
    is_running = False

    for i in range(len(valid_time)):
        if i > 0:
            step_dt = valid_time[i] - valid_time[i-1]
            # If there is a large gap in data, the run breaks
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
            
    # 7. Stability Analysis during the Max Contiguous Phase
    stability_score = 0.0
    if max_contiguous_duration > 0:
        # Map valid indices back to original signal indices to get CoP
        # Note: active_mask aligns signals to valid_area.
        # We need CoP for the "on toes" segment.
        
        # Get indices in the original arrays corresponding to the active mask
        active_indices = np.where(active_mask)[0]
        
        # Slice the "on toes" run from the active indices
        if max_run_end_idx > max_run_start_idx:
            run_indices = active_indices[max_run_start_idx : max_run_end_idx+1]
            
            run_cop_x = signals.cop_x[run_indices]
            run_cop_y = signals.cop_y[run_indices]
            
            # Calculate sway speed during this specific run
            dx = np.diff(run_cop_x)
            dy = np.diff(run_cop_y)
            run_dt = dt # Approximation
            speed = np.sqrt(dx**2 + dy**2) / run_dt
            
            # Filter extreme noise
            valid_speed = speed[speed < 500]
            if len(valid_speed) > 0:
                stability_score = float(np.mean(valid_speed))
    
    # Calculate Flat Foot Stability (Baseline Stability)
    # Map 'is_on_toes' (which is only for active frames) back to full length
    is_on_toes_full = np.zeros_like(active_mask, dtype=bool)
    is_on_toes_full[active_mask] = is_on_toes
    
    # Use indices where active but NOT on toes
    flat_mask = active_mask & (~is_on_toes_full)
    
    flat_global_indices = np.where(flat_mask)[0]
    
    baseline_sway_speed = 0.0
    if len(flat_global_indices) > 10:
        # Speed calculation for full signal
        dx = np.diff(signals.cop_x)
        dy = np.diff(signals.cop_y)
        full_speed = np.sqrt(dx**2 + dy**2) / dt
        full_speed = np.concatenate(([0], full_speed))
        
        # Extract speeds during flat phase
        flat_speeds = full_speed[flat_global_indices]
        # Filter
        flat_speeds = flat_speeds[(flat_speeds < 500) & np.isfinite(flat_speeds)]
        if len(flat_speeds) > 0:
            baseline_sway_speed = float(np.mean(flat_speeds))
            
    # Adaptive Instability Threshold
    # If they are stable flat (e.g., 5-10), threshold will be ~30-40.
    # If they are unstable flat (e.g., 30), threshold will be ~75.
    base_thresh = 40.0
    if baseline_sway_speed > 0:
        instability_threshold = max(base_thresh, baseline_sway_speed * 2.5)
    else:
        instability_threshold = 50.0 # Fallback
    
    if max_contiguous_duration < 3.0:
        score = 0
    else:
        # Duration is good. Check stability against adaptive threshold.
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
        "_valid_time": valid_time,         # Pass for plotting
        "_smoothed_area": smoothed_area,   # Pass for plotting
        "_on_toes_mask": is_on_toes        # Pass for plotting
    }

    return ExerciseResult(
        participant=participant,
        exercise="Rise to toes",
        variant=None,
        file_path=csv_path,
        score=score,
        features=features,
    )


def process_sit_to_stand(csv_path: str, participant: str, used_hands: bool = False, multiple_attempts: bool = False) -> ExerciseResult:
    """
    Sit to Stand.
    Scoring:
    (2) Normal: Comes to stand without use of hands and stabilizes independently.
    (1) Moderate: Comes to stand WITH use of hands on first attempt.
    (0) Severe: Unable to stand up from chair without assistance, OR needs several attempts with use of hands.
    """
    signals = load_basic_signals(csv_path)
    full_duration = _duration(signals)
    
    f = signals.force
    time_s = signals.time_s
    
    # Detect Rise Phase using Total Force
    # Force should go from low (seated, feet maybe on mat but low weight) to high (standing, full weight)
    # Or if they start seated OFF mat, force goes 0 -> BodyWeight
    # If they start seated ON mat (chair on mat?), force is BodyWeight -> BodyWeight + Momentum -> BodyWeight.
    # Assuming chair is OFF mat, feet on mat. Force increases as they shift weight to feet.
    
    max_f = np.max(f)
    
    # Smooth force for robust detection
    window = max(int(0.5 / signals.dt), 3)
    f_smooth = np.convolve(f, np.ones(window)/window, mode='same')
    
    # Thresholds for detection
    # Stand start: Force exceeds 20% of max (weight shift begins)
    # Stand stable: Force reaches 90% of max and stays there
    
    start_thresh = 0.2 * max_f
    stable_thresh = 0.85 * max_f
    
    # Find start
    start_indices = np.where(f_smooth > start_thresh)[0]
    if len(start_indices) > 0:
        start_idx = start_indices[0]
        start_time = float(time_s[start_idx])
    else:
        start_idx = 0
        start_time = 0.0
        
    # Find stable (end of rise)
    # Look for when force stays high consistently
    stable_indices = np.where(f_smooth > stable_thresh)[0]
    if len(stable_indices) > 0:
        # Simple approach: first time it crosses stable threshold
        end_idx = stable_indices[0]
        end_time = float(time_s[end_idx])
        
        # Calculate rise duration
        rise_duration = end_time - start_time
    else:
        end_time = float("nan")
        rise_duration = 0.0
        
    # Scoring Logic
    if multiple_attempts:
        score = 0
    else:
        # Check if they actually stood up (force increased significantly)
        # If max force is very low (< 20), they probably didn't stand on the mat
        if max_f < 20:
            score = 0
        elif used_hands:
            score = 1
        else:
            # No hands, single attempt. Did they stabilize?
            # We assume if force curve is reasonable, they stabilized.
            # Refine: check if force drops back to 0 (sat down / fell)?
            # For now, assume success if force went up and clinician didn't mark multiple attempts.
            score = 2
        
    features = {
        "Rise Time (s)": rise_duration,
        "Rise Start Time (s)": start_time,
        "Rise End Time (s)": end_time,
        "Max Force": float(max_f),
        "Used Hands": used_hands,
        "Multiple Attempts": multiple_attempts
    }

    return ExerciseResult(
        participant=participant,
        exercise="Sit to Stand",
        variant=None,
        file_path=csv_path,
        score=score,
        features=features,
    )


def process_stand_on_one_leg(csv_path: str, participant: str, stance_leg: str) -> ExerciseResult:
    """
    Stand on one leg.
    """
    signals = load_basic_signals(csv_path)
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)

    if duration >= 20.0:
        score = 2
    elif duration >= 10.0:
        score = 1
    else:
        score = 0

    features = {
        "Duration (s)": duration,
        "CoP Path Length": sway.get("cop_path_length", float("nan")),
        "CoP RMS": sway.get("cop_rms", float("nan")),
    }

    return ExerciseResult(
        participant=participant,
        exercise="Stand on one leg",
        variant=stance_leg,
        file_path=csv_path,
        score=score,
        features=features,
    )


def process_stance_eyes_open(csv_path: str, participant: str) -> ExerciseResult:
    """Stance feet together, eyes open."""
    signals = load_basic_signals(csv_path)
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    stance_metrics = _stance_balance_metrics(signals)
    
    # --- New Step Detection Logic ---
    # A step is a failure of the stance. The effective duration is time until the first step.
    step_count, first_step_time = _track_step_count_cv(signals)
    
    if np.isnan(first_step_time):
        # No step was detected, so the person stood for the full duration
        effective_stance_duration = duration
    else:
        # A step was detected, so the stance ended at that time
        effective_stance_duration = first_step_time

    # --- New Scoring Logic ---
    if effective_stance_duration >= 30.0:
        score = 2 # Normal: stood for 30s without stepping
    elif effective_stance_duration >= 2.0:
        score = 1 # Moderate: stood for at least 2s, but stepped before 30s
    else:
        score = 0 # Severe: unable to hold stance for even 2 seconds

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

    return ExerciseResult(
        participant=participant,
        exercise="Stance feet together (eyes open, firm)",
        variant=None,
        file_path=csv_path,
        score=score,
        features=features,
    )


def process_stance_eyes_closed(csv_path: str, participant: str) -> ExerciseResult:
    """Stance feet together, eyes closed."""
    signals = load_basic_signals(csv_path)
    duration = _duration(signals)
    sway = _sway_metrics(signals.cop_x, signals.cop_y)
    stance_metrics = _stance_balance_metrics(signals)

    # --- New Step Detection Logic ---
    # A step is a failure of the stance. The effective duration is time until the first step.
    step_count, first_step_time = _track_step_count_cv(signals)

    if np.isnan(first_step_time):
        # No step was detected, so the person stood for the full duration
        effective_stance_duration = duration
    else:
        # A step was detected, so the stance ended at that time
        effective_stance_duration = first_step_time

    # --- New Scoring Logic ---
    if effective_stance_duration >= 30.0:
        score = 2 # Normal: stood for 30s without stepping
    elif effective_stance_duration >= 2.0:
        score = 1 # Moderate: stood for at least 2s, but stepped before 30s
    else:
        score = 0 # Severe: unable to hold stance for even 2 seconds


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

    return ExerciseResult(
        participant=participant,
        exercise="Stance feet together (eyes closed, foam)",
        variant=None,
        file_path=csv_path,
        score=score,
        features=features,
    )


def _infer_participant_and_exercise(root: str, filename: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Infer participant, exercise, and variant (direction / side) from filename.
    """
    participant = os.path.basename(root)
    name = filename

    # Compensatory stepping
    if "Compensatory_stepping_correction-_BACKWARD" in name:
        return participant, "comp_stepping_backward", "BACKWARD"
    if "Compensatory_stepping_correction-_FORWARD" in name:
        return participant, "comp_stepping_forward", "FORWARD"
    if "Compensatory_stepping_correction-_LATERAL" in name and "RIGHT SIDE" in name:
        return participant, "comp_stepping_lateral_right", "LATERAL-RIGHT"
    if "Compensatory_stepping_correction-_LATERAL" in name and "LEFT SIDE" in name:
        return participant, "comp_stepping_lateral_left", "LATERAL-LEFT"

    # Rise to toes
    if "Rise_to_toes" in name:
        return participant, "rise_to_toes", None

    # Sit to stand
    if "Sit_to_stand" in name:
        return participant, "sit_to_stand", None

    # Stance eyes open / closed
    if "Stance,_Eyes_open" in name:
        return participant, "stance_eyes_open", None
    if "Stance,_Eyes_closed" in name:
        return participant, "stance_eyes_closed", None

    # Stand on one leg
    if "Stand_on_one_leg" in name and "RIGHT LEG OFF" in name:
        # Right leg off -> standing on LEFT
        return participant, "stand_one_leg_left", "Left"
    if "Stand_on_one_leg" in name and "LEFT LEG OFF" in name:
        # Left leg off -> standing on RIGHT
        return participant, "stand_one_leg_right", "Right"

    return participant, None, None


def process_all(data_root: str = ".") -> List[ExerciseResult]:
    """
    Walk through Evita / Kostas / Panos folders, process all requested exercises,
    and return a list of ExerciseResult objects.
    """
    results: List[ExerciseResult] = []

    for participant in ["Evita", "Kostas", "Panos"]:
        participant_dir = os.path.join(data_root, participant)
        if not os.path.isdir(participant_dir):
            continue

        for root, _, files in os.walk(participant_dir):
            for fname in files:
                if not fname.lower().endswith(".csv"):
                    continue

                part, ex_key, variant = _infer_participant_and_exercise(root, fname)
                if ex_key is None:
                    continue

                csv_path = os.path.join(root, fname)

                try:
                    if ex_key.startswith("comp_stepping"):
                        direction = variant or "UNKNOWN"
                        res = process_compensatory_stepping(csv_path, direction, part)
                    elif ex_key == "rise_to_toes":
                        res = process_rise_to_toes(csv_path, part)
                    elif ex_key == "sit_to_stand":
                        res = process_sit_to_stand(csv_path, part)
                    elif ex_key == "stance_eyes_open":
                        res = process_stance_eyes_open(csv_path, part)
                    elif ex_key == "stance_eyes_closed":
                        res = process_stance_eyes_closed(csv_path, part)
                    elif ex_key == "stand_one_leg_left":
                        res = process_stand_on_one_leg(csv_path, part, stance_leg="Left")
                    elif ex_key == "stand_one_leg_right":
                        res = process_stand_on_one_leg(csv_path, part, stance_leg="Right")
                    else:
                        continue

                    results.append(res)
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
                    import traceback
                    traceback.print_exc()

    return results


def results_to_dataframe(results: List[ExerciseResult]) -> pd.DataFrame:
    """Flatten ExerciseResult objects into a pandas DataFrame."""
    rows = []
    for r in results:
        base = {
            "participant": r.participant,
            "exercise": r.exercise,
            "variant": r.variant,
            "file_path": r.file_path,
            "score": r.score,
        }
        base.update(r.features)
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    results = process_all(".")
    if not results:
        print("No exercise CSVs found for Evita / Kostas / Panos.")
        return

    df = results_to_dataframe(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("Per-file scores:")
    print(df[["participant", "exercise", "variant", "score", "file_path"]])

    print("\nMean scores per participant and exercise:")
    summary = df.groupby(["participant", "exercise", "variant"])["score"].mean().reset_index()
    print(summary)


if __name__ == "__main__":
    main()
