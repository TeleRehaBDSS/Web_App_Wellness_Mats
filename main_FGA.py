import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ast
from backend.analyze_data_functions_MAT import (
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


@dataclass
class FGAResult:
    """Result of FGA exercise analysis."""
    participant: str
    exercise: str
    file_path: str
    score: int                  # FGA score 0-3
    features: Dict[str, any]
    explanation: str


def load_fga_signals(csv_path: str) -> FGASignals:
    """Load FGA data from CSV with 6 mats."""
    df = pd.read_csv(csv_path)
    df['timepoint'] = pd.to_datetime(df['timepoint'])
    df = df.sort_values(['sample', 'mat']).reset_index(drop=True)
    
    # Get unique samples (time frames)
    unique_samples = sorted(df['sample'].unique())
    
    # Generate combined matrices for each sample
    frames = []
    timepoints = []
    
    for sample in unique_samples:
        rows = df[df['sample'] == sample]
        combined_mat = combine_mats(rows)
        frames.append(combined_mat)
        timepoints.append(rows.iloc[0]['timepoint'])
    
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
        df=df
    )


def process_fga_gait_level_surface(csv_path: str, participant: str) -> FGAResult:
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
        return FGAResult(
            participant=participant,
            exercise="Gait Level Surface",
            file_path=csv_path,
            score=0,
            features={},
            explanation="No gait cycles detected. Cannot complete 6m walk."
        )
    
    # Calculate total distance walked (approximate from mat positions)
    # Each mat is approximately 48 cm wide, 6 mats = ~288 cm = ~2.88 m
    # We need to estimate if they walked 6m (20 ft)
    # Assuming each mat is ~48 cm, 6 mats ≈ 2.88 m
    # For 6m walk, we'd need approximately 12.5 mats worth of distance
    
    # Calculate time to walk
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
    
    # Estimate distance walked (in cm)
    # Use horizontal distance from CoP traces or step positions
    number_of_steps = len(structured_steps)
    
    # Calculate deviation from walkway center
    # Walkway width is 30.48 cm (12 in), so center is at 15.24 cm from each edge
    # We need to measure lateral deviation from the centerline
    
    # Extract CoP x positions (lateral/medial) for deviation calculation
    cop_x_positions = []
    for trace in signals.cop_traces:
        if 'cop_x' in trace:
            cop_x_positions.extend(trace['cop_x'])
    
    if cop_x_positions:
        cop_x_array = np.array(cop_x_positions)
        # Assuming mat width is 48 units, walkway center is at 24 units per mat
        # For 6 mats, center is at 144 units (48*6/2)
        walkway_center = 144  # Center of 6 mats
        deviations = np.abs(cop_x_array - walkway_center)
        max_deviation_cm = np.max(deviations) * (30.48 / 48) if len(deviations) > 0 else 0  # Convert to cm
        average_deviation_cm = np.mean(deviations) * (30.48 / 48) if len(deviations) > 0 else 0
    else:
        max_deviation_cm = 0
        average_deviation_cm = 0
    
    # Estimate if they walked 6m
    # If we have stride lengths, estimate total distance
    if average_stride_length > 0 and number_of_steps > 0:
        estimated_distance_cm = average_stride_length * number_of_steps
        estimated_distance_m = estimated_distance_cm / 100
    else:
        # Fallback: estimate from time and cadence
        # Average walking speed ~1.2 m/s
        estimated_distance_m = total_time * 1.2 if total_time > 0 else 0
    
    # Calculate time to walk 6m (if we can estimate)
    # If estimated_distance_m > 0, scale time proportionally
    if estimated_distance_m > 0:
        time_to_walk_6m = (6.0 / estimated_distance_m) * total_time
    else:
        time_to_walk_6m = total_time  # Use actual time as fallback
    
    # Check for assistive device (would need manual input, assume False for now)
    uses_assistive_device = False  # TODO: Add manual input option
    
    # Grading according to FGA.pdf
    score = 0
    explanation_parts = []
    
    # Check if they can walk 6m
    if estimated_distance_m < 4.0:  # Less than ~4m suggests inability
        score = 0
        explanation_parts.append("Cannot walk 6m without assistance or severe impairment.")
    elif time_to_walk_6m < 5.5 and max_deviation_cm <= 15.24 and not uses_assistive_device and average_cadence >= 50:
        score = 3
        explanation_parts.append(
            f"Normal: Walks 6m in {time_to_walk_6m:.2f}s (< 5.5s), "
            f"max deviation {max_deviation_cm:.1f}cm (≤ 15.24cm), "
            f"no assistive device, good speed (cadence: {average_cadence:.1f} steps/min)."
        )
    elif 5.5 <= time_to_walk_6m < 7.0 and max_deviation_cm <= 25.4:
        score = 2
        explanation_parts.append(
            f"Mild impairment: Walks 6m in {time_to_walk_6m:.2f}s (5.5-7s), "
            f"max deviation {max_deviation_cm:.1f}cm (≤ 25.4cm), "
            f"slower speed or mild gait deviations."
        )
    elif time_to_walk_6m >= 7.0 or max_deviation_cm <= 38.1:
        score = 1
        explanation_parts.append(
            f"Moderate impairment: Walks 6m in {time_to_walk_6m:.2f}s (≥ 7s) "
            f"or max deviation {max_deviation_cm:.1f}cm (≤ 38.1cm), "
            f"slow speed, abnormal gait pattern, or evidence of imbalance."
        )
    else:
        score = 0
        explanation_parts.append(
            f"Severe impairment: Cannot walk 6m without assistance, "
            f"max deviation {max_deviation_cm:.1f}cm (> 38.1cm), "
            f"or severe gait deviations/imbalance."
        )
    
    explanation = " ".join(explanation_parts)
    
    features = {
        "time_to_walk_6m_s": time_to_walk_6m,
        "total_time_s": total_time,
        "estimated_distance_m": estimated_distance_m,
        "number_of_steps": number_of_steps,
        "average_cadence_steps_per_min": average_cadence,
        "average_stride_length_cm": average_stride_length,
        "max_deviation_cm": max_deviation_cm,
        "average_deviation_cm": average_deviation_cm,
        "uses_assistive_device": uses_assistive_device,
        "gait_cycles": gait_cycles,
        "structured_steps": structured_steps,
        "_cop_traces": signals.cop_traces,
        "_time_s": signals.time_s.tolist(),
    }
    
    return FGAResult(
        participant=participant,
        exercise="Gait Level Surface",
        file_path=csv_path,
        score=score,
        features=features,
        explanation=explanation
    )

