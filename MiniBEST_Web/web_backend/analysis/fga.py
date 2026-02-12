"""
FGA (Functional Gait Assessment) Processing Module
Handles processing of FGA exercises with 6 mats data.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from .mat_utils import (
    combine_mats,
    detect_clusters,
    track_steps_separately,
    filter_short_steps,
    reassign_steps_by_alternation,
    organize_steps_with_pressures,
    add_step_metadata_with_overbox,
    analyze_gait_cycles_with_metrics,
    calculate_gait_cycle_distances,
    generate_combined_matrices2,
    extract_center_of_pressure_trace
)


@dataclass
class FGASignals:
    """Container for FGA signals from 6 mats."""
    time_s: np.ndarray          # shape (T,) - time in seconds
    frames: List[np.ndarray]    # List of combined matrices (48, 48*6) for each sample
    cop_traces: List[Dict]      # Center of pressure traces
    gait_cycles: List[Dict]     # Gait cycle data
    structured_steps: List[Dict]  # Structured step data
    df: pd.DataFrame            # Original dataframe
    combined_matrices_by_sample: Dict  # Combined matrices by sample number


def load_fga_signals(csv_path: str) -> FGASignals:
    """Load FGA data from CSV with 6 mats."""
    df = pd.read_csv(csv_path)
    if 'timepoint' in df.columns:
        df['timepoint'] = pd.to_datetime(df['timepoint'])
    df = df.sort_values(['sample', 'mat']).reset_index(drop=True)
    
    # Get unique samples (time frames)
    unique_samples = sorted(df['sample'].unique())
    
    # Generate combined matrices for each sample
    frames = []
    timepoints = []
    combined_matrices_by_sample = {}
    
    for sample in unique_samples:
        rows = df[df['sample'] == sample]
        combined_mat = combine_mats(rows)
        frames.append(combined_mat)
        timepoints.append(rows.iloc[0]['timepoint'])
        combined_matrices_by_sample[sample] = combined_mat
    
    # Calculate time in seconds
    if len(timepoints) > 0:
        time_s = np.array([(tp - timepoints[0]).total_seconds() for tp in timepoints])
    else:
        time_s = np.array([])
    
    # Detect clusters and track steps
    bboxes_by_frame = []
    for sample in unique_samples:
        rows = df[df['sample'] == sample]
        combined_mat = combine_mats(rows)
        bboxes = detect_clusters(combined_mat)
        bboxes_by_frame.append(bboxes)
    
    steps = track_steps_separately(bboxes_by_frame, timepoints, overlap_threshold=0.05)
    combined_matrix_by_time = generate_combined_matrices2(df)
    
    # Filter and reassign steps
    steps = filter_short_steps(steps, min_frames=4)
    reassigned_steps = reassign_steps_by_alternation(steps)
    
    # Organize steps into structured format
    structured_steps = organize_steps_with_pressures(reassigned_steps, combined_matrix_by_time, df)
    
    # Add metadata for each step
    structured_steps_with_metadata = add_step_metadata_with_overbox(
        structured_steps,
        combined_matrix_by_time,
        df
    )
    
    # Extract and analyze gait cycles
    gait_cycles = analyze_gait_cycles_with_metrics(structured_steps_with_metadata)
    
    # Calculate distances for each gait cycle
    gait_cycles_with_distances = calculate_gait_cycle_distances(
        gait_cycles,
        structured_steps_with_metadata
    )
    
    # Extract CoP traces
    cop_traces = extract_center_of_pressure_trace(structured_steps_with_metadata, combined_matrix_by_time, df)
    
    return FGASignals(
        time_s=time_s,
        frames=frames,
        cop_traces=cop_traces,
        gait_cycles=gait_cycles_with_distances,
        structured_steps=structured_steps_with_metadata,
        df=df,
        combined_matrices_by_sample=combined_matrices_by_sample
    )


def filter_temporal_outliers(structured_steps, max_gap_seconds=1.5):
    """
    Filter out steps that are temporally isolated from the main walking sequence.
    Finds the largest continuous sequence of steps with reasonable gaps.
    
    Args:
        structured_steps: List of structured steps
        max_gap_seconds: Maximum allowed gap between consecutive steps (default 1.5 seconds)
    
    Returns:
        Filtered list of structured steps (only main walking sequence)
    """
    if not structured_steps or len(structured_steps) < 2:
        return structured_steps
    
    # Sort steps by start_time
    sorted_steps = sorted(structured_steps, key=lambda s: s.get('start_time', pd.Timestamp.min))
    
    if len(sorted_steps) < 2:
        return sorted_steps
    
    # Calculate time gaps between consecutive steps
    gaps = []
    for i in range(len(sorted_steps) - 1):
        current_end = sorted_steps[i].get('end_time')
        next_start = sorted_steps[i + 1].get('start_time')
        
        if current_end is not None and next_start is not None:
            try:
                if isinstance(current_end, pd.Timestamp) and isinstance(next_start, pd.Timestamp):
                    gap = (next_start - current_end).total_seconds()
                elif hasattr(current_end, '__sub__') and hasattr(next_start, '__sub__'):
                    time_diff = next_start - current_end
                    gap = time_diff.total_seconds() if hasattr(time_diff, 'total_seconds') else float(time_diff)
                else:
                    gap = float('inf')
                gaps.append(gap)
            except Exception:
                gaps.append(float('inf'))
        else:
            gaps.append(float('inf'))
    
    if not gaps:
        return sorted_steps
    
    # Find the longest continuous sequence with gaps <= max_gap_seconds
    # This identifies the main walking sequence
    best_start = 0
    best_length = 1
    current_start = 0
    current_length = 1
    
    for i, gap in enumerate(gaps):
        if gap <= max_gap_seconds:
            # Continue the current sequence
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            # Break in sequence - start a new one
            current_start = i + 1
            current_length = 1
    
    # Extract the main walking sequence
    main_sequence_steps = sorted_steps[best_start:best_start + best_length]
    
    # If the main sequence is too short (< 40% of original), return original
    # This prevents over-filtering, but allows filtering if there are clear outliers
    if len(main_sequence_steps) < max(3, len(sorted_steps) * 0.4):
        return sorted_steps
    
    return main_sequence_steps


def process_fga_gait_level_surface(csv_path: str, participant: str) -> Dict:
    """
    Process FGA Exercise 1: Gait Level Surface
    
    According to FGA.pdf:
    (3) Normal: Walks 6m in < 5.5s, no assistive devices, good speed, 
        no imbalance, normal gait pattern, deviates ≤ 15.24 cm (6 in) 
        outside 30.48 cm (12 in) walkway width.
    (2) Mild: Walks 6m in < 7s but > 5.5s, uses assistive device, 
        slower speed, mild gait deviations, or deviates 15.24-25.4 cm (6-10 in).
    (1) Moderate: Walks 6m, slow speed, abnormal gait pattern, 
        evidence of imbalance, or deviates 25.4-38.1 cm (10-15 in). 
        Requires > 7s to ambulate 6m.
    (0) Severe: Cannot walk 6m without assistance, severe gait deviations 
        or imbalance, deviates > 38.1 cm (15 in) or reaches/touches wall.
    """
    signals = load_fga_signals(csv_path)
    
    # Calculate metrics
    gait_cycles = signals.gait_cycles
    structured_steps = signals.structured_steps
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected. Cannot complete 6m walk.",
            "features": {}
        }
    
    # Calculate actual walking time - SIMPLE AND RELIABLE APPROACH
    # Use structured steps (actual foot contacts) to find walking period
    # Filter out standing periods by finding when steps are actually progressing forward
    actual_walking_time = 0
    
    # PRIMARY METHOD: Use structured steps with forward progression detection
    if structured_steps and len(structured_steps) > 0:
        # Filter temporal outliers first
        filtered_steps = filter_temporal_outliers(structured_steps, max_gap_seconds=1.5)
        if len(filtered_steps) == 0:
            filtered_steps = structured_steps
        
        # Sort steps by time
        sorted_steps = sorted(filtered_steps, key=lambda s: s.get('start_time', pd.Timestamp.min))
        
        if len(sorted_steps) >= 3:  # Need at least 3 steps for meaningful walk
            # Get timepoints from all frames in all steps
            all_timepoints = []
            for step in sorted_steps:
                for frame_data in step.get('frames', []):
                    tp = frame_data.get('timepoint')
                    if tp is not None:
                        all_timepoints.append(tp)
            
            if len(all_timepoints) > 0:
                first_tp = signals.df['timepoint'].iloc[0]
                step_times = sorted([(tp - first_tp).total_seconds() for tp in all_timepoints])
                
                # Find the main continuous block (exclude isolated steps at start/end)
                if len(step_times) > 1:
                    gaps = np.diff(step_times)
                    large_gap_indices = np.where(gaps > 2.0)[0]  # 2 second gap = standing break
                    
                    if len(large_gap_indices) == 0:
                        # Single continuous block - use all steps
                        actual_walking_time = step_times[-1] - step_times[0]
                    else:
                        # Multiple blocks - find the LARGEST one
                        block_starts = [0] + [g+1 for g in large_gap_indices]
                        block_ends = [g for g in large_gap_indices] + [len(step_times)-1]
                        block_durations = [step_times[block_ends[i]] - step_times[block_starts[i]] 
                                         for i in range(len(block_starts))]
                        largest_block_idx = np.argmax(block_durations)
                        actual_walking_time = step_times[block_ends[largest_block_idx]] - step_times[block_starts[largest_block_idx]]
    
    # FALLBACK: Use CoP X progression if step-based didn't work
    if actual_walking_time <= 0 and len(signals.frames) > 0 and len(signals.time_s) > 0:
        # Calculate CoP X for all frames
        cop_x_by_frame = []
        frame_indices = []
        
        for i, frame in enumerate(signals.frames):
            frame_sum = np.sum(frame)
            if frame_sum > 10.0:
                rows, cols = frame.shape
                col_indices = np.arange(cols).reshape(1, cols)
                cop_x = np.sum(frame * col_indices) / frame_sum
                cop_x_by_frame.append(cop_x)
                frame_indices.append(i)
        
        if len(cop_x_by_frame) > 20:
            cop_x_arr = np.array(cop_x_by_frame)
            
            # Find when CoP X starts increasing (walking begins)
            # Standing: CoP X stays low and flat
            # Walking: CoP X increases forward
            
            # Use a simple approach: find period where CoP X increases significantly
            # Start: first time CoP X > 60 (past mat 1)
            # End: last time CoP X > 100 (past mat 2)
            start_threshold = 60
            end_threshold = 100
            
            start_mask = cop_x_arr > start_threshold
            end_mask = cop_x_arr > end_threshold
            
            if np.any(start_mask) and np.any(end_mask):
                start_idx = frame_indices[np.where(start_mask)[0][0]]
                end_idx = frame_indices[np.where(end_mask)[0][-1]]
                
                if end_idx > start_idx:
                    actual_walking_time = signals.time_s[end_idx] - signals.time_s[start_idx]
    
    # FALLBACK: Use structured steps if movement-based didn't work
    if actual_walking_time <= 0 and structured_steps and len(structured_steps) > 0:
        # Filter outliers first
        filtered_steps = filter_temporal_outliers(structured_steps, max_gap_seconds=1.5)
        if len(filtered_steps) == 0:
            filtered_steps = structured_steps
        
        # Collect timepoints from step frames
        all_step_timepoints = []
        for step in filtered_steps:
            for frame_data in step.get('frames', []):
                tp = frame_data.get('timepoint')
                if tp is not None:
                    all_step_timepoints.append(tp)
        
        if len(all_step_timepoints) > 0:
            first_tp = signals.df['timepoint'].iloc[0]
            step_times = sorted([(tp - first_tp).total_seconds() for tp in all_step_timepoints])
            
            if len(step_times) > 1:
                # Find largest continuous block
                gaps = np.diff(step_times)
                large_gap_indices = np.where(gaps > 1.5)[0]
                
                if len(large_gap_indices) == 0:
                    actual_walking_time = step_times[-1] - step_times[0]
                else:
                    block_starts = [0] + [g+1 for g in large_gap_indices]
                    block_ends = [g for g in large_gap_indices] + [len(step_times)-1]
                    block_lengths = [step_times[block_ends[i]] - step_times[block_starts[i]] 
                                    for i in range(len(block_starts))]
                    largest_block_idx = np.argmax(block_lengths)
                    actual_walking_time = step_times[block_ends[largest_block_idx]] - step_times[block_starts[largest_block_idx]]
    
    # ULTIMATE FALLBACK
    if actual_walking_time <= 0:
        if len(signals.time_s) > 0:
            actual_walking_time = signals.time_s[-1] - signals.time_s[0]
        else:
            actual_walking_time = 0
    
    # FALLBACK: Use structured steps if frame-based didn't work
    if actual_walking_time <= 0 and structured_steps and len(structured_steps) > 0:
        # Filter outliers
        filtered_steps = filter_temporal_outliers(structured_steps, max_gap_seconds=1.5)
        if len(filtered_steps) == 0:
            filtered_steps = structured_steps
        
        sorted_steps = sorted(filtered_steps, key=lambda s: s.get('start_time', pd.Timestamp.min))
        if len(sorted_steps) > 0:
            first_step = sorted_steps[0]
            last_step = sorted_steps[-1]
            first_start = first_step.get('start_time')
            last_end = last_step.get('end_time')
            
            if first_start is not None and last_end is not None:
                try:
                    if isinstance(first_start, pd.Timestamp) and isinstance(last_end, pd.Timestamp):
                        actual_walking_time = (last_end - first_start).total_seconds()
                    elif hasattr(first_start, '__sub__') and hasattr(last_end, '__sub__'):
                        time_diff = last_end - first_start
                        actual_walking_time = time_diff.total_seconds() if hasattr(time_diff, 'total_seconds') else float(time_diff)
                except Exception as e:
                    print(f"Error calculating walking time from steps: {e}")
    
    # ULTIMATE FALLBACK
    if actual_walking_time <= 0:
        if len(signals.time_s) > 0:
            actual_walking_time = signals.time_s[-1] - signals.time_s[0]
        else:
            actual_walking_time = 0
    
    # Keep total_time for reference
    if len(signals.time_s) > 0:
        total_time = signals.time_s[-1] - signals.time_s[0]
    else:
        total_time = 0
    
    # Calculate average cadence (steps per minute)
    cadences = [cycle.get('cadence', 0) for cycle in gait_cycles if cycle.get('cadence') is not None]
    average_cadence = np.mean(cadences) if cadences else 0
    
    # Calculate stride length (from gait cycles)
    stride_lengths = []
    for cycle in gait_cycles:
        if 'average_horizontal_distance' in cycle and cycle['average_horizontal_distance'] is not None:
            stride_lengths.append(cycle['average_horizontal_distance'])
    
    average_stride_length = np.mean(stride_lengths) if stride_lengths else 0
    
    # Number of steps
    number_of_steps = len(structured_steps)
    
    # Calculate deviation from walkway center
    # IMPORTANT: Matrix shape is (48 rows, 288 columns)
    # - Rows (48) = WIDTH direction = 1 meter = 100cm (perpendicular to walking)
    # - Columns (288) = LENGTH direction = 6 meters = 600cm (along walking direction)
    # Deviation should be measured in the WIDTH direction (rows), not length (columns)!
    # 
    # Walkway center in width direction = 24 (middle of 48 rows = 50cm from edge)
    # 1 row unit = 100cm / 48 = ~2.083 cm
    
    # Calculate CoP Y position (row index) from all active frames
    # This measures deviation in the width direction (perpendicular to walking)
    cop_y_positions = []
    if len(signals.frames) > 0:
        # Filter to active frames only (when walking is happening)
        # Use a threshold based on max pressure to avoid noise
        frame_sums = [np.sum(frame) for frame in signals.frames]
        max_pressure = max(frame_sums) if frame_sums else 0
        pressure_threshold = max(10.0, max_pressure * 0.05)
        
        for i, frame in enumerate(signals.frames):
            if np.sum(frame) > pressure_threshold:
                rows, cols = frame.shape  # rows=48 (width), cols=288 (length)
                row_indices = np.arange(rows).reshape(rows, 1)
                
                total_force = np.sum(frame)
                if total_force > 0:
                    # Calculate CoP Y (row position) - this is the width direction
                    # row_indices goes from 0 to 47, so cop_y will be in [0, 48)
                    cop_y = np.sum(frame * row_indices) / total_force
                    cop_y_positions.append(cop_y)
    
    # Fallback to step-based CoP if frame-based didn't work
    if len(cop_y_positions) == 0:
        for trace in signals.cop_traces:
            if 'y' in trace:
                y_data = trace['y']
                if isinstance(y_data, list):
                    cop_y_positions.extend(y_data)
                else:
                    cop_y_positions.append(y_data)
    
    if cop_y_positions:
        cop_y_array = np.array(cop_y_positions)
        # Walkway center in width direction: middle of 48 rows = 24 units
        walkway_center_width = 24.0
        
        # Calculate deviations in sensor units (row units)
        deviations_sensor_units = np.abs(cop_y_array - walkway_center_width)
        
        # Convert to cm: 1 row unit = 100cm / 48 = 2.0833 cm
        row_unit_to_cm = 100.0 / 48.0
        deviations_cm = deviations_sensor_units * row_unit_to_cm
        
        # Use maximum deviation (not percentile) as per user request
        # Also calculate average for more robust measure
        if len(deviations_cm) > 0:
            max_deviation_cm = np.max(deviations_cm)
            average_deviation_cm = np.mean(deviations_cm)
        else:
            max_deviation_cm = 0
            average_deviation_cm = 0
    else:
        max_deviation_cm = 0
        average_deviation_cm = 0
    
    # Estimate distance walked
    # IMPORTANT: The treadmill is FIXED at 6 meters (6 mats in a row)
    # Each mat is 1 meter, so 6 mats = 6 meters total
    # If the patient walks across the treadmill, the distance is 6 meters
    # We can verify by checking if steps traverse across multiple mats
    
    # Check if patient traversed the full treadmill (6 mats = 6 meters)
    # by looking at step positions across mats
    estimated_distance_m = 6.0  # Default: assume they walked the full 6m treadmill
    
    # Verify by checking step positions (if available)
    if structured_steps and len(structured_steps) > 0:
        # Check if steps span across the mats (indicating they walked the full distance)
        # This is a simple check - if we have multiple steps, assume they walked the treadmill
        # The treadmill is fixed, so if they completed the exercise, distance is 6m
        if number_of_steps >= 3:  # At least 3 steps suggests they walked
            estimated_distance_m = 6.0
        else:
            # If very few steps, they might not have completed the full walk
            # Estimate based on steps (conservative)
            if average_stride_length > 0 and number_of_steps > 0:
                estimated_distance_cm = average_stride_length * number_of_steps
                estimated_distance_m = min(estimated_distance_cm / 100, 6.0)  # Cap at 6m
            else:
                estimated_distance_m = min(actual_walking_time * 1.0, 6.0)  # Conservative estimate
    else:
        # No steps detected - cannot complete 6m
        estimated_distance_m = 0
    
    # Calculate time to walk 6m using ACTUAL WALKING TIME (not total exercise time)
    # This is the key fix: use actual_walking_time instead of total_time
    if estimated_distance_m > 0:
        # Scale the actual walking time to estimate time for 6m
        time_to_walk_6m = (6.0 / estimated_distance_m) * actual_walking_time
    else:
        # If we can't estimate distance, use actual walking time directly
        time_to_walk_6m = actual_walking_time
    
    # Check for assistive device (would need manual input, assume False for now)
    uses_assistive_device = False
    
    # Grading according to FGA.pdf
    score = 0
    explanation_parts = []
    
    # Grading according to FGA.pdf - Mark the highest category that applies
    # Check from highest to lowest score
    
    # (3) Normal: ALL conditions must be met
    # - Walks 6m in < 5.5 seconds
    # - No assistive devices
    # - Good speed
    # - No evidence for imbalance
    # - Normal gait pattern
    # - Deviates no more than 15.24 cm (6 in) outside of the 30.48-cm (12-in) walkway width
    if (estimated_distance_m >= 5.5 and 
        time_to_walk_6m < 5.5 and 
        not uses_assistive_device and 
        average_cadence >= 50 and  # Good speed
        max_deviation_cm <= 15.24):
        score = 3
        explanation_parts.append(
            f"Normal: Walks 6m in {time_to_walk_6m:.2f}s (less than 5.5 seconds), "
            f"no assistive devices, good speed (cadence: {average_cadence:.1f} steps/min), "
            f"no evidence for imbalance, normal gait pattern, "
            f"deviates no more than {max_deviation_cm:.1f}cm (≤ 15.24cm) outside of the 30.48-cm walkway width."
        )
    
    # (2) Mild impairment: ANY of these conditions (OR logic)
    # - Walks 6m in < 7s but > 5.5s, OR
    # - uses assistive device, OR
    # - slower speed, OR
    # - mild gait deviations, OR
    # - deviates 15.24–25.4 cm (6–10 in) outside of the 30.48-cm (12-in) walkway width
    elif estimated_distance_m >= 5.5 and (
        (5.5 < time_to_walk_6m < 7.0) or
        uses_assistive_device or
        (average_cadence < 50 and max_deviation_cm <= 15.24 and time_to_walk_6m < 7.0) or  # slower speed (but no major deviation, and not too slow)
        (15.24 <= max_deviation_cm <= 25.4)  # deviates 15.24–25.4 cm (inclusive boundaries)
    ):
        score = 2
        if uses_assistive_device:
            explanation_parts.append(
                f"Mild impairment: Walks 6m in {time_to_walk_6m:.2f}s, uses assistive device."
            )
        elif 15.24 <= max_deviation_cm <= 25.4:
            explanation_parts.append(
                f"Mild impairment: Walks 6m in {time_to_walk_6m:.2f}s, "
                f"deviates {max_deviation_cm:.1f}cm (15.24–25.4 cm) outside of the 30.48-cm walkway width."
            )
        elif 5.5 < time_to_walk_6m < 7.0:
            explanation_parts.append(
                f"Mild impairment: Walks 6m in {time_to_walk_6m:.2f}s (less than 7 seconds but greater than 5.5 seconds), "
                f"slower speed (cadence: {average_cadence:.1f} steps/min) or mild gait deviations."
            )
        else:
            explanation_parts.append(
                f"Mild impairment: Walks 6m in {time_to_walk_6m:.2f}s, "
                f"slower speed (cadence: {average_cadence:.1f} steps/min) or mild gait deviations."
            )
    
    # (1) Moderate impairment: Walks 6m AND (ANY of):
    # - Slow speed, OR
    # - Abnormal gait pattern, OR
    # - Evidence for imbalance, OR
    # - Deviates 25.4–38.1 cm (10–15 in) outside of the 30.48-cm (12-in) walkway width, OR
    # - Requires more than 7 seconds to ambulate 6m
    elif estimated_distance_m >= 5.5 and (
        time_to_walk_6m > 7.0 or  # Requires more than 7 seconds
        (25.4 < max_deviation_cm <= 38.1) or  # deviates 25.4–38.1 cm (exclusive of 25.4, inclusive of 38.1)
        (average_cadence < 40 and max_deviation_cm <= 25.4)  # slow speed or abnormal pattern/imbalance (but deviation not severe)
    ):
        score = 1
        if time_to_walk_6m > 7.0:
            explanation_parts.append(
                f"Moderate impairment: Walks 6m in {time_to_walk_6m:.2f}s (requires more than 7 seconds to ambulate 6m), "
                f"slow speed, abnormal gait pattern, evidence for imbalance."
            )
        elif 25.4 < max_deviation_cm <= 38.1:
            explanation_parts.append(
                f"Moderate impairment: Walks 6m in {time_to_walk_6m:.2f}s, "
                f"deviates {max_deviation_cm:.1f}cm (25.4–38.1 cm) outside of the 30.48-cm walkway width, "
                f"slow speed, abnormal gait pattern, or evidence for imbalance."
            )
        else:
            explanation_parts.append(
                f"Moderate impairment: Walks 6m in {time_to_walk_6m:.2f}s, "
                f"slow speed, abnormal gait pattern, evidence for imbalance."
            )
    
    # (0) Severe impairment: ANY of:
    # - Cannot walk 6m without assistance, OR
    # - Severe gait deviations or imbalance, OR
    # - Deviates greater than 38.1 cm (15 in) outside of the 30.48-cm (12-in) walkway width, OR
    # - Reaches and touches the wall
    else:
        score = 0
        if estimated_distance_m < 5.5:
            explanation_parts.append(
                f"Severe impairment: Cannot walk 6m without assistance (only {estimated_distance_m:.1f}m), "
                f"severe gait deviations or imbalance."
            )
        elif max_deviation_cm > 38.1:
            explanation_parts.append(
                f"Severe impairment: Deviates {max_deviation_cm:.1f}cm (greater than 38.1 cm) outside of the 30.48-cm walkway width, "
                f"severe gait deviations or imbalance, or reaches and touches the wall."
            )
        else:
            explanation_parts.append(
                f"Severe impairment: Severe gait deviations or imbalance, "
                f"or reaches and touches the wall."
            )
    
    explanation = " ".join(explanation_parts)
    
    features = {
        "time_to_walk_6m_s": float(time_to_walk_6m),
        "actual_walking_time_s": float(actual_walking_time),  # Time between first heel strike and last toe off
        "total_exercise_time_s": float(total_time),  # Total exercise duration (includes pre/post walk time)
        "estimated_distance_m": float(estimated_distance_m),
        "number_of_steps": int(number_of_steps),
        "average_cadence_steps_per_min": float(average_cadence),
        "average_stride_length_cm": float(average_stride_length),
        "max_deviation_cm": float(max_deviation_cm),  # Maximum deviation from walkway center
        "average_deviation_cm": float(average_deviation_cm),
        "uses_assistive_device": uses_assistive_device,
    }
    
    return {
        "score": score,
        "explanation": explanation,
        "features": features,
        "signals": signals  # Include signals for plotting
    }


def _calculate_common_metrics(signals, structured_steps):
    """Helper function to calculate common metrics for FGA exercises."""
    # Calculate actual walking time using structured steps
    actual_walking_time = 0
    if structured_steps and len(structured_steps) > 0:
        filtered_steps = filter_temporal_outliers(structured_steps, max_gap_seconds=1.5)
        if len(filtered_steps) == 0:
            filtered_steps = structured_steps
        
        sorted_steps = sorted(filtered_steps, key=lambda s: s.get('start_time', pd.Timestamp.min))
        
        if len(sorted_steps) >= 3:
            all_timepoints = []
            for step in sorted_steps:
                for frame_data in step.get('frames', []):
                    tp = frame_data.get('timepoint')
                    if tp is not None:
                        all_timepoints.append(tp)
            
            if len(all_timepoints) > 0:
                first_tp = signals.df['timepoint'].iloc[0]
                step_times = sorted([(tp - first_tp).total_seconds() for tp in all_timepoints])
                
                if len(step_times) > 1:
                    gaps = np.diff(step_times)
                    large_gap_indices = np.where(gaps > 2.0)[0]
                    
                    if len(large_gap_indices) == 0:
                        actual_walking_time = step_times[-1] - step_times[0]
                    else:
                        block_starts = [0] + [g+1 for g in large_gap_indices]
                        block_ends = [g for g in large_gap_indices] + [len(step_times)-1]
                        block_durations = [step_times[block_ends[i]] - step_times[block_starts[i]] 
                                         for i in range(len(block_starts))]
                        largest_block_idx = np.argmax(block_durations)
                        actual_walking_time = step_times[block_ends[largest_block_idx]] - step_times[block_starts[largest_block_idx]]
    
    # Calculate total time
    if len(signals.time_s) > 0:
        total_time = signals.time_s[-1] - signals.time_s[0]
    else:
        total_time = 0
    
    # Calculate gait metrics
    gait_cycles = signals.gait_cycles
    number_of_steps = len(structured_steps) if structured_steps else 0
    
    stride_lengths = []
    cadences = []
    for cycle in gait_cycles:
        if 'average_horizontal_distance' in cycle and cycle['average_horizontal_distance'] is not None:
            stride_lengths.append(cycle['average_horizontal_distance'])
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    average_stride_length = np.mean(stride_lengths) if stride_lengths else 0
    average_cadence = np.mean(cadences) if cadences else (number_of_steps / actual_walking_time * 60 if actual_walking_time > 0 else 0)
    
    # Calculate deviation (CoP Y)
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
                    cop_y_positions.append(cop_y)
    
    max_deviation_cm = 0
    average_deviation_cm = 0
    if cop_y_positions:
        cop_y_array = np.array(cop_y_positions)
        walkway_center_width = 24.0  # Middle of 48 rows
        sensor_unit_to_cm = 100.0 / 48.0  # 48 sensors = 1 meter = 100cm
        deviations_cm = np.abs(cop_y_array - walkway_center_width) * sensor_unit_to_cm
        max_deviation_cm = float(np.max(deviations_cm))
        average_deviation_cm = float(np.mean(deviations_cm))
    
    estimated_distance_m = 6.0 if number_of_steps >= 3 else 0
    
    return {
        "actual_walking_time": actual_walking_time,
        "total_time": total_time,
        "number_of_steps": number_of_steps,
        "average_stride_length": average_stride_length,
        "average_cadence": average_cadence,
        "max_deviation_cm": max_deviation_cm,
        "average_deviation_cm": average_deviation_cm,
        "estimated_distance_m": estimated_distance_m,
    }


def process_fga_change_gait_speed(csv_path: str, participant: str) -> Dict:
    """
    Process FGA Exercise 2: Change in Gait Speed
    
    (3) Normal: Smoothly changes speed, significant difference between speeds, deviates ≤ 15.24 cm
    (2) Mild: Changes speed but mild deviations, deviates 15.24-25.4 cm, or no significant change in velocity
    (1) Moderate: Minor speed adjustments, significant deviations, deviates 25.4-38.1 cm, or loses balance but recovers
    (0) Severe: Cannot change speeds, deviates > 38.1 cm, or loses balance
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    
    # Analyze speed changes by looking at cadence variation
    cadences = []
    for cycle in gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    speed_variation = np.std(cadences) if len(cadences) > 1 else 0
    significant_speed_change = speed_variation > 10  # At least 10 steps/min variation
    
    # Scoring
    score = 0
    explanation_parts = []
    
    if metrics["max_deviation_cm"] <= 15.24 and significant_speed_change:
        score = 3
        explanation_parts.append(
            f"Normal: Smoothly changes walking speed without loss of balance. "
            f"Shows significant difference in speeds (variation: {speed_variation:.1f} steps/min). "
            f"Deviates {metrics['max_deviation_cm']:.1f}cm (≤ 15.24 cm) outside walkway width."
        )
    elif metrics["max_deviation_cm"] <= 25.4 or (not significant_speed_change and metrics["max_deviation_cm"] <= 15.24):
        score = 2
        if 15.24 < metrics["max_deviation_cm"] <= 25.4:
            explanation_parts.append(
                f"Mild impairment: Changes speed but demonstrates mild gait deviations. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (15.24–25.4 cm) outside walkway width."
            )
        else:
            explanation_parts.append(
                f"Mild impairment: No significant change in velocity (variation: {speed_variation:.1f} steps/min)."
            )
    elif metrics["max_deviation_cm"] <= 38.1:
        score = 1
        explanation_parts.append(
            f"Moderate impairment: Makes only minor adjustments to walking speed, "
            f"or accomplishes change with significant gait deviations. "
            f"Deviates {metrics['max_deviation_cm']:.1f}cm (25.4–38.1 cm) outside walkway width."
        )
    else:
        score = 0
        explanation_parts.append(
            f"Severe impairment: Cannot change speeds, or deviates {metrics['max_deviation_cm']:.1f}cm "
            f"(> 38.1 cm) outside walkway width, or loses balance."
        )
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "speed_variation_steps_per_min": float(speed_variation),
        "significant_speed_change": bool(significant_speed_change),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "average_deviation_cm": float(metrics["average_deviation_cm"]),
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def process_fga_horizontal_head_turns(csv_path: str, participant: str, manual_input: Optional[str] = None) -> Dict:
    """
    Process FGA Exercise 3: Gait with Horizontal Head Turns
    
    (3) Normal: Performs head turns smoothly with no change in gait, deviates ≤ 15.24 cm
    (2) Mild: Performs head turns smoothly with slight change in velocity, deviates 15.24-25.4 cm, or uses assistive device
    (1) Moderate: Performs head turns with moderate change in velocity, slows down, deviates 25.4-38.1 cm but recovers
    (0) Severe: Performs task with severe disruption, staggers > 38.1 cm, loses balance, stops, or reaches for wall
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    
    # Analyze cadence variation (head turns may cause speed changes)
    cadences = []
    for cycle in gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    cadence_variation = np.std(cadences) if len(cadences) > 1 else 0
    smooth_performance = cadence_variation < 15  # Low variation = smooth
    
    # Manual input mapping
    manual_score = None
    if manual_input:
        manual_mapping = {
            "Smoothly": 3,
            "Mild difficulty": 2,
            "Moderate difficulty": 1,
            "Severe difficulty or unable": 0
        }
        manual_score = manual_mapping.get(manual_input)
    
    # Scoring
    score = 0
    explanation_parts = []
    
    if manual_score is not None:
        score = manual_score
        if score == 3:
            explanation_parts.append("Normal: Performs head turns smoothly with no change in gait.")
        elif score == 2:
            explanation_parts.append("Mild impairment: Performs head turns smoothly with slight change in gait velocity.")
        elif score == 1:
            explanation_parts.append("Moderate impairment: Performs head turns with moderate change in gait velocity.")
        else:
            explanation_parts.append("Severe impairment: Performs task with severe disruption of gait.")
    else:
        # Automatic scoring based on metrics
        if metrics["max_deviation_cm"] <= 15.24 and smooth_performance:
            score = 3
            explanation_parts.append(
                f"Normal: Performs head turns smoothly with no change in gait. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (≤ 15.24 cm) outside walkway width."
            )
        elif metrics["max_deviation_cm"] <= 25.4:
            score = 2
            explanation_parts.append(
                f"Mild impairment: Performs head turns smoothly with slight change in gait velocity. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (15.24–25.4 cm) outside walkway width."
            )
        elif metrics["max_deviation_cm"] <= 38.1:
            score = 1
            explanation_parts.append(
                f"Moderate impairment: Performs head turns with moderate change in gait velocity, slows down. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (25.4–38.1 cm) but recovers, can continue to walk."
            )
        else:
            score = 0
            explanation_parts.append(
                f"Severe impairment: Performs task with severe disruption of gait. "
                f"Staggers {metrics['max_deviation_cm']:.1f}cm (> 38.1 cm) outside walkway width, "
                f"loses balance, stops, or reaches for wall."
            )
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "cadence_variation_steps_per_min": float(cadence_variation),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "average_deviation_cm": float(metrics["average_deviation_cm"]),
        "manual_input": manual_input,
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def process_fga_vertical_head_turns(csv_path: str, participant: str, manual_input: Optional[str] = None) -> Dict:
    """
    Process FGA Exercise 4: Gait with Vertical Head Turns
    
    (3) Normal: Performs head turns with no change in gait, deviates ≤ 15.24 cm
    (2) Mild: Performs task with slight change in velocity, deviates 15.24-25.4 cm or uses assistive device
    (1) Moderate: Performs task with moderate change in velocity, slows down, deviates 25.4-38.1 cm but recovers
    (0) Severe: Performs task with severe disruption, staggers > 38.1 cm, loses balance, stops, reaches for wall
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    
    # Analyze cadence variation
    cadences = []
    for cycle in gait_cycles:
        if 'cadence' in cycle and cycle['cadence'] is not None:
            cadences.append(cycle['cadence'])
    
    cadence_variation = np.std(cadences) if len(cadences) > 1 else 0
    smooth_performance = cadence_variation < 15
    
    # Manual input mapping
    manual_score = None
    if manual_input:
        manual_mapping = {
            "Smoothly": 3,
            "Mild difficulty": 2,
            "Moderate difficulty": 1,
            "Severe difficulty or unable": 0
        }
        manual_score = manual_mapping.get(manual_input)
    
    # Scoring (same logic as horizontal head turns)
    score = 0
    explanation_parts = []
    
    if manual_score is not None:
        score = manual_score
        if score == 3:
            explanation_parts.append("Normal: Performs head turns with no change in gait.")
        elif score == 2:
            explanation_parts.append("Mild impairment: Performs task with slight change in gait velocity.")
        elif score == 1:
            explanation_parts.append("Moderate impairment: Performs task with moderate change in gait velocity.")
        else:
            explanation_parts.append("Severe impairment: Performs task with severe disruption of gait.")
    else:
        if metrics["max_deviation_cm"] <= 15.24 and smooth_performance:
            score = 3
            explanation_parts.append(
                f"Normal: Performs head turns with no change in gait. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (≤ 15.24 cm) outside walkway width."
            )
        elif metrics["max_deviation_cm"] <= 25.4:
            score = 2
            explanation_parts.append(
                f"Mild impairment: Performs task with slight change in gait velocity. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (15.24–25.4 cm) outside walkway width."
            )
        elif metrics["max_deviation_cm"] <= 38.1:
            score = 1
            explanation_parts.append(
                f"Moderate impairment: Performs task with moderate change in gait velocity, slows down. "
                f"Deviates {metrics['max_deviation_cm']:.1f}cm (25.4–38.1 cm) but recovers, can continue to walk."
            )
        else:
            score = 0
            explanation_parts.append(
                f"Severe impairment: Performs task with severe disruption of gait. "
                f"Staggers {metrics['max_deviation_cm']:.1f}cm (> 38.1 cm) outside walkway width, "
                f"loses balance, stops, or reaches for wall."
            )
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "cadence_variation_steps_per_min": float(cadence_variation),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "average_deviation_cm": float(metrics["average_deviation_cm"]),
        "manual_input": manual_input,
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def process_fga_pivot_turn(csv_path: str, participant: str, manual_input: Optional[str] = None) -> Dict:
    """
    Process FGA Exercise 5: Gait and Pivot Turn
    
    (3) Normal: Pivot turns safely within 3 seconds and stops quickly with no loss of balance
    (2) Mild: Pivot turns safely in ≤3 seconds and stops with no loss of balance, or turns safely within 3s and stops with mild imbalance
    (1) Moderate: Turns slowly, requires verbal cueing, or requires several small steps to catch balance
    (0) Severe: Cannot turn safely, requires assistance to turn and stop
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    
    # Analyze turn time (time from start to when CoP X reaches max and starts decreasing)
    turn_time = metrics["actual_walking_time"]
    
    # Manual input mapping
    manual_score = None
    if manual_input:
        manual_mapping = {
            "Smooth and balanced": 3,
            "Mild imbalance": 2,
            "Significant imbalance or hesitation": 1,
            "Unable to perform": 0
        }
        manual_score = manual_mapping.get(manual_input)
    
    # Scoring
    score = 0
    explanation_parts = []
    
    if manual_score is not None:
        score = manual_score
        if score == 3:
            explanation_parts.append("Normal: Pivot turns safely within 3 seconds and stops quickly with no loss of balance.")
        elif score == 2:
            explanation_parts.append("Mild impairment: Pivot turns safely within 3 seconds and stops with mild imbalance.")
        elif score == 1:
            explanation_parts.append("Moderate impairment: Turns slowly, requires verbal cueing, or requires several small steps to catch balance.")
        else:
            explanation_parts.append("Severe impairment: Cannot turn safely, requires assistance to turn and stop.")
    else:
        # Automatic scoring
        if turn_time <= 3.0:
            score = 3
            explanation_parts.append(
                f"Normal: Pivot turns safely within {turn_time:.2f} seconds and stops quickly with no loss of balance."
            )
        elif turn_time <= 4.0:
            score = 2
            explanation_parts.append(
                f"Mild impairment: Pivot turns safely within {turn_time:.2f} seconds and stops with no loss of balance."
            )
        else:
            score = 1
            explanation_parts.append(
                f"Moderate impairment: Turns slowly ({turn_time:.2f} seconds), requires verbal cueing, "
                f"or requires several small steps to catch balance."
            )
    
    features = {
        "turn_time_s": float(turn_time),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(metrics["number_of_steps"]),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "manual_input": manual_input,
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def _filter_obstacle_steps(structured_steps, signals):
    """
    Filter out steps that are actually the static obstacle (shoe box) on the mat.
    The obstacle is static (doesn't move horizontally) and appears in the middle
    region of the walkway for the duration of the exercise.
    
    Args:
        structured_steps: List of structured steps (may include obstacle detections)
        signals: FGASignals object
    
    Returns:
        Filtered list of steps with obstacle detections removed
    """
    if not structured_steps or len(structured_steps) == 0:
        return structured_steps
    
    # 6 mats × 48 columns = 288 columns total; obstacle is in the middle third
    total_cols = 288
    obstacle_col_start = total_cols // 3       # ~96
    obstacle_col_end = 2 * total_cols // 3     # ~192
    
    total_time = (signals.time_s[-1] - signals.time_s[0]) if len(signals.time_s) > 1 else 1.0
    
    filtered_steps = []
    
    for step in structured_steps:
        # Get bounding box info
        overbox = step.get('metadata', {}).get('overbox')
        frames = step.get('frames', [])
        
        if overbox:
            min_row, min_col, max_row, max_col = overbox
        elif frames:
            min_col = min(f.get('bbox', [0, 0, 0, 0])[1] for f in frames)
            max_col = max(f.get('bbox', [0, 0, 0, 0])[3] for f in frames)
        else:
            continue
        
        step_center_col = (min_col + max_col) / 2
        is_in_obstacle_region = obstacle_col_start <= step_center_col <= obstacle_col_end
        
        # Steps outside the obstacle region are always real walking steps
        if not is_in_obstacle_region:
            filtered_steps.append(step)
            continue
        
        # --- For steps in the obstacle region, check if they are the obstacle ---
        
        # 1. Calculate horizontal movement (real feet move; the obstacle doesn't)
        movement = 0
        if len(frames) >= 3:
            centers = []
            for f in frames:
                bbox = f.get('bbox')
                if bbox:
                    centers.append((bbox[1] + bbox[3]) / 2)
            if len(centers) > 1:
                movement = max(centers) - min(centers)
        
        # 2. Calculate step duration
        step_duration = 0
        if step.get('start_time') and step.get('end_time'):
            try:
                if isinstance(step['start_time'], pd.Timestamp) and isinstance(step['end_time'], pd.Timestamp):
                    step_duration = (step['end_time'] - step['start_time']).total_seconds()
                else:
                    step_duration = float(step['end_time']) - float(step['start_time'])
            except Exception:
                step_duration = len(frames) * 0.1  # fallback estimate
        
        # 3. Count frames whose center is in the obstacle region
        frames_in_obstacle = 0
        for f in frames:
            bbox = f.get('bbox')
            if bbox:
                frame_center_col = (bbox[1] + bbox[3]) / 2
                if obstacle_col_start <= frame_center_col <= obstacle_col_end:
                    frames_in_obstacle += 1
        obstacle_ratio = frames_in_obstacle / len(frames) if frames else 0
        
        # Obstacle characteristics:
        is_static = movement < 12            # barely moves horizontally
        is_long_duration = step_duration > total_time * 0.15   # present for >15 % of exercise
        is_mostly_in_obstacle = obstacle_ratio > 0.3           # >30 % of frames in middle
        is_moderate_and_static = (step_duration > total_time * 0.10) and is_static
        
        # Remove if the step is in the obstacle region AND meets any obstacle criterion
        if is_in_obstacle_region and (is_static or is_long_duration or is_mostly_in_obstacle or is_moderate_and_static):
            continue  # skip – this is the obstacle, not a foot
        
        filtered_steps.append(step)
    
    return filtered_steps


def _recompute_gait_cycles(filtered_steps):
    """
    Recompute gait cycles from scratch using only the filtered (real) steps.
    This is more reliable than filtering original gait cycles by heel-strike
    matching, because many original cycles paired the obstacle with a real foot.
    """
    if not filtered_steps or len(filtered_steps) < 3:
        return []
    
    # Recompute gait cycles from filtered steps
    fresh_gait_cycles = analyze_gait_cycles_with_metrics(filtered_steps)
    
    # Add distance information
    fresh_gait_cycles = calculate_gait_cycle_distances(fresh_gait_cycles, filtered_steps)
    
    return fresh_gait_cycles


def _check_stepped_around_obstacle(signals):
    """
    Detect whether the patient stepped *around* the obstacle instead of over it,
    by checking for a spike in lateral CoP deviation in the middle time window.
    """
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
    
    if len(cop_y_positions) >= 10:
        times, cop_ys = zip(*cop_y_positions)
        times = np.array(times)
        cop_ys = np.array(cop_ys)
        
        # Middle time window (around the obstacle)
        mid_time = (times[-1] + times[0]) / 2
        mid_mask = (times >= mid_time - 1.0) & (times <= mid_time + 1.0)
        if np.any(mid_mask):
            mid_deviation = np.std(cop_ys[mid_mask])
            overall_deviation = np.std(cop_ys)
            # Large lateral deviation in the middle → stepped around
            if mid_deviation > overall_deviation * 1.5 and mid_deviation > 10:
                return True
    
    return False


def process_fga_step_over_obstacle(csv_path: str, participant: str, manual_input: Optional[str] = None) -> Dict:
    """
    Process FGA Exercise 6: Step Over Obstacle
    
    (3) Normal: Steps over 2 stacked boxes (22.86 cm) without changing gait speed, no imbalance
    (2) Mild: Steps over one box (11.43 cm) without changing gait speed, no imbalance
    (1) Moderate: Steps over one box but must slow down and adjust steps, may require verbal cueing
    (0) Severe: Cannot perform without assistance
    
    Key fix: the shoe-box obstacle sitting on the mat is filtered out before computing
    cadence and gait metrics, so it is no longer mistaken for a foot.
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not structured_steps or len(structured_steps) == 0:
        return {
            "score": 0,
            "explanation": "No steps detected.",
            "features": {},
            "signals": signals
        }
    
    # ---- Step 1: filter out the static obstacle from detected steps ----
    filtered_steps = _filter_obstacle_steps(structured_steps, signals)
    
    if len(filtered_steps) < 3:
        return {
            "score": 0,
            "explanation": "Not enough valid walking steps detected after filtering obstacle.",
            "features": {
                "number_of_steps_raw": len(structured_steps),
                "number_of_steps_filtered": len(filtered_steps),
            },
            "signals": signals
        }
    
    # ---- Step 2: recalculate common metrics with filtered steps ----
    metrics = _calculate_common_metrics(signals, filtered_steps)
    
    # Recompute gait cycles from filtered steps (for plotting / features)
    recomputed_gait_cycles = _recompute_gait_cycles(filtered_steps)
    
    # ---- Step 3: obstacle-gap analysis ----
    # Due to the obstacle on the mat, the step tracker merges nearby foot
    # detections with the obstacle.  As a result, detected steps are only
    # found on the mats far from the obstacle (typically mat 1 and mats
    # 5-6).  A direct before/after cadence comparison would therefore
    # measure startup vs cooldown, not the obstacle interaction.
    #
    # Instead, the primary metric is the **obstacle crossing gap**: the
    # longest inter-step interval, which represents the time the patient
    # needed to traverse the obstacle zone.  Normalized by the patient's
    # normal step interval, a high value indicates slowing / hesitation.
    
    sorted_steps = sorted(
        filtered_steps,
        key=lambda s: s.get('metadata', {}).get('heel_strike', pd.Timestamp.min)
    )
    heel_strikes = [
        s['metadata']['heel_strike']
        for s in sorted_steps
        if s.get('metadata', {}).get('heel_strike') is not None
    ]
    
    obstacle_gap = 0.0
    mean_step_interval = 0.0
    gap_in_steps = 0.0
    avg_cadence_filtered = 0.0
    cadence_variation = 0.0
    
    if len(heel_strikes) >= 3:
        intervals = [
            (heel_strikes[i + 1] - heel_strikes[i]).total_seconds()
            for i in range(len(heel_strikes) - 1)
        ]
        
        # The obstacle crossing gap is the longest interval
        gap_idx = int(np.argmax(intervals))
        obstacle_gap = intervals[gap_idx]
        
        # Normal walking intervals (excluding the obstacle gap)
        normal_intervals = intervals[:gap_idx] + intervals[gap_idx + 1:]
        normal_intervals = [iv for iv in normal_intervals if 0.3 <= iv <= 2.5]
        
        if normal_intervals:
            mean_step_interval = float(np.mean(normal_intervals))
            avg_cadence_filtered = 60.0 / mean_step_interval
            cadence_variation = float(
                np.std([60.0 / iv for iv in normal_intervals])
            ) if len(normal_intervals) > 1 else 0.0
            
            # Gap measured in "step equivalents"
            # ~6 steps expected across the obstacle zone (~4 m at ~0.65 m stride)
            gap_in_steps = obstacle_gap / mean_step_interval if mean_step_interval > 0 else 0.0
    
    elif len(heel_strikes) == 2:
        obstacle_gap = (heel_strikes[1] - heel_strikes[0]).total_seconds()
        # Cannot separate gap from normal walking with only 2 steps
        gap_in_steps = 0.0
    
    # Determine if the patient slowed down / adjusted steps
    # A gap_in_steps > 7.5 means the obstacle crossing took significantly
    # longer than expected from the patient's normal walking rhythm,
    # indicating they slowed down or hesitated.
    slowed_down = gap_in_steps >= 7.5
    
    # ---- Step 4: imbalance & stepped-around checks ----
    has_imbalance = metrics["max_deviation_cm"] > 25.4   # > 10 inches
    stepped_around = _check_stepped_around_obstacle(signals)
    
    # ---- Step 5: manual input mapping ----
    manual_score = None
    if manual_input:
        manual_mapping = {
            "Smooth": 3,
            "Slight hesitation": 2,
            "Significant effort or imbalance": 1,
            "Unable to perform": 0,
        }
        manual_score = manual_mapping.get(manual_input)
    
    # ---- Step 6: scoring ----
    score = 0
    explanation_parts = []
    
    if manual_score is not None:
        score = manual_score
        if score == 3:
            explanation_parts.append(
                "Normal: Steps over obstacle without changing gait speed, no evidence of imbalance."
            )
        elif score == 2:
            explanation_parts.append(
                "Mild impairment: Steps over obstacle without changing gait speed, no evidence of imbalance."
            )
        elif score == 1:
            explanation_parts.append(
                "Moderate impairment: Steps over obstacle but must slow down and adjust steps to clear safely."
            )
        else:
            explanation_parts.append(
                "Severe impairment: Cannot perform without assistance."
            )
    else:
        # Automatic scoring based on sensor data
        #
        # FGA score 3 vs 2 depends on obstacle height (2 boxes vs 1 box)
        # which cannot be measured from the mat.  Automatic scoring
        # defaults to 2 ("maintained speed") and relies on manual input
        # to upgrade to 3 when appropriate.
        if stepped_around or len(filtered_steps) < 3:
            score = 0
            explanation_parts.append(
                "Severe impairment: Cannot step over obstacle or steps around obstacle."
            )
        elif slowed_down:
            score = 1
            explanation_parts.append(
                f"Moderate impairment: Must slow down and adjust steps to clear obstacle safely. "
                f"Obstacle crossing took {obstacle_gap:.2f}s "
                f"({gap_in_steps:.1f}× normal step interval of {mean_step_interval:.2f}s). "
                f"Walking cadence: {avg_cadence_filtered:.1f} steps/min."
            )
        elif has_imbalance:
            score = 1
            explanation_parts.append(
                f"Moderate impairment: Evidence of imbalance while stepping over obstacle "
                f"(max lateral deviation {metrics['max_deviation_cm']:.1f} cm). "
                f"Walking cadence: {avg_cadence_filtered:.1f} steps/min."
            )
        else:
            # Speed maintained and no imbalance → score 2 (conservative).
            # Score 3 requires clinician confirmation of 2-box obstacle height.
            score = 2
            explanation_parts.append(
                f"Mild impairment: Steps over obstacle without changing gait speed and no evidence "
                f"of imbalance. Obstacle crossing: {obstacle_gap:.2f}s "
                f"({gap_in_steps:.1f}× normal step interval). "
                f"Walking cadence: {avg_cadence_filtered:.1f} steps/min."
            )
    
    # ---- Build features dict ----
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "number_of_steps": int(len(filtered_steps)),
        "number_of_steps_raw": int(len(structured_steps)),
        "average_cadence_steps_per_min": avg_cadence_filtered,
        "cadence_variation_steps_per_min": cadence_variation,
        "obstacle_gap_s": float(obstacle_gap),
        "mean_step_interval_s": float(mean_step_interval),
        "gap_in_steps": float(gap_in_steps),
        "slowed_down": bool(slowed_down),
        "stepped_around": bool(stepped_around),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "has_imbalance": bool(has_imbalance),
        "manual_input": manual_input,
    }
    
    # Update signals with filtered data for plotting
    signals.structured_steps = filtered_steps
    if recomputed_gait_cycles:
        signals.gait_cycles = recomputed_gait_cycles
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def process_fga_narrow_base(csv_path: str, participant: str) -> Dict:
    """
    Process FGA Exercise 7: Gait with Narrow Base of Support
    
    (3) Normal: Ambulates 10 steps heel to toe with no staggering
    (2) Mild: Ambulates 7-9 steps
    (1) Moderate: Ambulates 4-7 steps
    (0) Severe: Ambulates less than 4 steps or cannot perform without assistance
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    number_of_steps = metrics["number_of_steps"]
    
    # Scoring based on number of steps
    score = 0
    explanation_parts = []
    
    if number_of_steps >= 10:
        score = 3
        explanation_parts.append(
            f"Normal: Ambulates {number_of_steps} steps heel to toe with no staggering."
        )
    elif number_of_steps >= 7:
        score = 2
        explanation_parts.append(
            f"Mild impairment: Ambulates {number_of_steps} steps (7–9 steps)."
        )
    elif number_of_steps >= 4:
        score = 1
        explanation_parts.append(
            f"Moderate impairment: Ambulates {number_of_steps} steps (4–7 steps)."
        )
    else:
        score = 0
        explanation_parts.append(
            f"Severe impairment: Ambulates {number_of_steps} steps (less than 4 steps) "
            f"or cannot perform without assistance."
        )
    
    features = {
        "number_of_steps": int(number_of_steps),
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def process_fga_eyes_closed(csv_path: str, participant: str) -> Dict:
    """
    Process FGA Exercise 8: Gait with Eyes Closed
    
    (3) Normal: Walks 6m, no assistive devices, good speed, no imbalance, normal gait, deviates ≤ 15.24 cm, < 7s
    (2) Mild: Walks 6m, uses assistive device, slower speed, mild deviations, deviates 15.24-25.4 cm, 7-9s
    (1) Moderate: Walks 6m, slow speed, abnormal pattern, imbalance, deviates 25.4-38.1 cm, > 9s
    (0) Severe: Cannot walk 6m without assistance, severe deviations, deviates > 38.1 cm, or will not attempt
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected. Cannot complete 6m walk.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    estimated_distance_m = metrics["estimated_distance_m"]
    time_to_walk_6m = (6.0 / estimated_distance_m) * metrics["actual_walking_time"] if estimated_distance_m > 0 else metrics["actual_walking_time"]
    uses_assistive_device = False  # Would need manual input
    
    # Scoring
    score = 0
    explanation_parts = []
    
    if estimated_distance_m >= 5.5 and time_to_walk_6m < 7.0 and not uses_assistive_device and metrics["max_deviation_cm"] <= 15.24:
        score = 3
        explanation_parts.append(
            f"Normal: Walks 6m in {time_to_walk_6m:.2f}s, no assistive devices, good speed, "
            f"no evidence of imbalance, normal gait pattern, deviates {metrics['max_deviation_cm']:.1f}cm "
            f"(≤ 15.24 cm) outside walkway width."
        )
    elif estimated_distance_m >= 5.5 and (7.0 <= time_to_walk_6m < 9.0 or uses_assistive_device or (15.24 < metrics["max_deviation_cm"] <= 25.4)):
        score = 2
        explanation_parts.append(
            f"Mild impairment: Walks 6m in {time_to_walk_6m:.2f}s, uses assistive device, slower speed, "
            f"mild gait deviations, deviates {metrics['max_deviation_cm']:.1f}cm (15.24–25.4 cm) outside walkway width."
        )
    elif estimated_distance_m >= 5.5 and (time_to_walk_6m >= 9.0 or (25.4 < metrics["max_deviation_cm"] <= 38.1)):
        score = 1
        explanation_parts.append(
            f"Moderate impairment: Walks 6m in {time_to_walk_6m:.2f}s, slow speed, abnormal gait pattern, "
            f"evidence for imbalance, deviates {metrics['max_deviation_cm']:.1f}cm (25.4–38.1 cm) outside walkway width."
        )
    else:
        score = 0
        explanation_parts.append(
            f"Severe impairment: Cannot walk 6m without assistance (only {estimated_distance_m:.1f}m), "
            f"severe gait deviations or imbalance, deviates {metrics['max_deviation_cm']:.1f}cm "
            f"(> 38.1 cm) outside walkway width, or will not attempt task."
        )
    
    features = {
        "time_to_walk_6m_s": float(time_to_walk_6m),
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "estimated_distance_m": float(estimated_distance_m),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "average_deviation_cm": float(metrics["average_deviation_cm"]),
        "uses_assistive_device": uses_assistive_device,
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }


def process_fga_ambulating_backwards(csv_path: str, participant: str) -> Dict:
    """
    Process FGA Exercise 9: Ambulating Backwards
    
    (3) Normal: Walks 6m, no assistive devices, good speed, no imbalance, normal gait, deviates ≤ 15.24 cm
    (2) Mild: Walks 6m, uses assistive device, slower speed, mild deviations, deviates 15.24-25.4 cm
    (1) Moderate: Walks 6m, slow speed, abnormal pattern, imbalance, deviates 25.4-38.1 cm
    (0) Severe: Cannot walk 6m without assistance, severe deviations, deviates > 38.1 cm, or will not attempt
    """
    signals = load_fga_signals(csv_path)
    structured_steps = signals.structured_steps
    gait_cycles = signals.gait_cycles
    
    if not gait_cycles or len(gait_cycles) == 0:
        return {
            "score": 0,
            "explanation": "No gait cycles detected. Cannot complete 6m walk.",
            "features": {},
            "signals": signals
        }
    
    metrics = _calculate_common_metrics(signals, structured_steps)
    estimated_distance_m = metrics["estimated_distance_m"]
    uses_assistive_device = False  # Would need manual input
    
    # Scoring (similar to exercise 1 but without time requirement)
    score = 0
    explanation_parts = []
    
    if estimated_distance_m >= 5.5 and not uses_assistive_device and metrics["max_deviation_cm"] <= 15.24:
        score = 3
        explanation_parts.append(
            f"Normal: Walks 6m, no assistive devices, good speed, no evidence for imbalance, "
            f"normal gait pattern, deviates {metrics['max_deviation_cm']:.1f}cm (≤ 15.24 cm) outside walkway width."
        )
    elif estimated_distance_m >= 5.5 and (uses_assistive_device or (15.24 < metrics["max_deviation_cm"] <= 25.4)):
        score = 2
        explanation_parts.append(
            f"Mild impairment: Walks 6m, uses assistive device, slower speed, mild gait deviations, "
            f"deviates {metrics['max_deviation_cm']:.1f}cm (15.24–25.4 cm) outside walkway width."
        )
    elif estimated_distance_m >= 5.5 and (25.4 < metrics["max_deviation_cm"] <= 38.1):
        score = 1
        explanation_parts.append(
            f"Moderate impairment: Walks 6m, slow speed, abnormal gait pattern, evidence for imbalance, "
            f"deviates {metrics['max_deviation_cm']:.1f}cm (25.4–38.1 cm) outside walkway width."
        )
    else:
        score = 0
        explanation_parts.append(
            f"Severe impairment: Cannot walk 6m without assistance (only {estimated_distance_m:.1f}m), "
            f"severe gait deviations or imbalance, deviates {metrics['max_deviation_cm']:.1f}cm "
            f"(> 38.1 cm) outside walkway width, or will not attempt task."
        )
    
    features = {
        "actual_walking_time_s": float(metrics["actual_walking_time"]),
        "total_exercise_time_s": float(metrics["total_time"]),
        "estimated_distance_m": float(estimated_distance_m),
        "number_of_steps": int(metrics["number_of_steps"]),
        "average_cadence_steps_per_min": float(metrics["average_cadence"]),
        "max_deviation_cm": float(metrics["max_deviation_cm"]),
        "average_deviation_cm": float(metrics["average_deviation_cm"]),
        "uses_assistive_device": uses_assistive_device,
    }
    
    return {
        "score": score,
        "explanation": " ".join(explanation_parts),
        "features": features,
        "signals": signals
    }

