import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import label

def parse_and_rotate_sensor_data(sensor_data):
    """Convert string sensor data to a 48x48 numpy matrix and rotate 180 degrees."""
    mat = np.array(ast.literal_eval(sensor_data))
    return np.rot90(mat, 2)

def combine_mats(rows):
    """Stack 6 rotated mats into a single horizontal matrix."""
    combined = np.zeros((48, 48 * 6))  # Updated for 6 mats
    if isinstance(rows, pd.DataFrame):
        iterator = rows.iterrows()
    else:
        # Handle case where rows might be a list or other iterable if needed
        iterator = rows.iterrows()

    for _, row in iterator:
        try:
            mat_index = int(row['mat'])
            if not 0 <= mat_index < 6:  # Updated range
                # print(f"Invalid mat_index: {mat_index}")
                continue
            sensor_data = parse_and_rotate_sensor_data(row['sensors'])
            if sensor_data.shape != (48, 48):
                # print(f"Unexpected sensor_data shape: {sensor_data.shape}")
                continue
            combined[:, mat_index * 48:(mat_index + 1) * 48] = sensor_data
        except Exception as e:
            # print(f"Error processing row: {e}")
            continue
    # if combined.sum() == 0:
    #     print("Warning: Combined matrix is empty.")
    return combined

# Merge bounding boxes aligned horizontally
def merge_bounding_boxes_horizontal(bboxes, horizontal_threshold=5):
    merged_bboxes = []
    for bbox in sorted(bboxes, key=lambda x: x[1]):  # Sort by column position
        min_row1, min_col1, max_row1, max_col1 = bbox
        merged = False
        for i, existing_bbox in enumerate(merged_bboxes):
            min_row2, min_col2, max_row2, max_col2 = existing_bbox
            vertical_overlap = (min_row1 <= max_row2) and (max_row1 >= min_row2)
            horizontal_proximity = (min_col1 - max_col2) <= horizontal_threshold and (min_col1 > max_col2)
            if vertical_overlap and horizontal_proximity:
                merged_bboxes[i] = (
                    min(min_row1, min_row2),
                    min(min_col1, min_col2),
                    max(max_row1, max_row2),
                    max(max_col1, max_col2),
                )
                merged = True
                break
        if not merged:
            merged_bboxes.append(bbox)
    return merged_bboxes

# Detect clusters and merge horizontally aligned ones
def detect_clusters(matrix, threshold=5, horizontal_threshold=5, min_size=(2, 2)):
    """Detect clusters of active sensors and merge horizontally aligned bounding boxes."""
    binary_matrix = (matrix > threshold).astype(int)
    labeled_matrix, num_clusters = label(binary_matrix)
    bboxes = []

    for cluster_id in range(1, num_clusters + 1):
        cluster_coords = np.argwhere(labeled_matrix == cluster_id)
        min_row, min_col = cluster_coords.min(axis=0)
        max_row, max_col = cluster_coords.max(axis=0)

        # Calculate the size of the bounding box
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        # Filter out small bounding boxes
        if height >= min_size[0] and width >= min_size[1]:
            bboxes.append((min_row, min_col, max_row, max_col))

    # Merge horizontally aligned bounding boxes
    merged_bboxes = merge_bounding_boxes_horizontal(bboxes, horizontal_threshold)
    return merged_bboxes

def calculate_overlap(bbox1, bbox2):
    min_row1, min_col1, max_row1, max_col1 = bbox1
    min_row2, min_col2, max_row2, max_col2 = bbox2
    overlap_row = max(0, min(max_row1, max_row2) - max(min_row1, min_row2))
    overlap_col = max(0, min(max_col1, max_col2) - max(min_col1, min_col2))
    overlap_area = overlap_row * overlap_col
    bbox1_area = (max_row1 - min_row1) * (max_col1 - min_col1)
    bbox2_area = (max_row2 - min_row2) * (max_col2 - min_col2)
    return overlap_area / max(bbox1_area, bbox2_area) if max(bbox1_area, bbox2_area) > 0 else 0

def calculate_bbox_area(bbox):
    """Calculate the area of a bounding box."""
    min_row, min_col, max_row, max_col = bbox
    return (max_row - min_row + 1) * (max_col - min_col + 1)

def filter_bboxes(bboxes, max_bboxes=2):
    """
    Filter bounding boxes to keep the largest ones by area.
    max_bboxes: Maximum number of bounding boxes to keep.
    """
    if len(bboxes) <= max_bboxes:
        return bboxes
    # Sort bboxes by area in descending order and keep the largest ones
    sorted_bboxes = sorted(bboxes, key=calculate_bbox_area, reverse=True)
    return sorted_bboxes[:max_bboxes]

def track_steps_separately(bboxes_by_frame, timepoints, overlap_threshold=0.05):
    """
    Track steps for left and right feet separately.
    Each step is a collection of successive bounding boxes assigned to either foot.
    """
    steps = {'left': [], 'right': []}  # Final steps
    active_left_steps = []  # Active steps for left foot
    active_right_steps = []  # Active steps for right foot
    # Track which foot is active initially
    left_foot_active = True
    right_foot_active = True
    for frame_idx, bboxes in enumerate(bboxes_by_frame):
        frame_timepoint = timepoints[frame_idx]

        # Filter bboxes to keep only the largest two
        bboxes = filter_bboxes(bboxes, max_bboxes=2)
        
        updated_left_steps = [False] * len(active_left_steps)
        updated_right_steps = [False] * len(active_right_steps)

        for bbox in bboxes:
            # Check overlap with left foot active steps
            best_overlap_left = 0
            best_left_idx = -1
            for i, step in enumerate(active_left_steps):
                step_bbox = step[-1]['bbox']  # Last bbox in the current active step
                overlap = calculate_overlap(bbox, step_bbox)
                if overlap > best_overlap_left:
                    best_overlap_left = overlap
                    best_left_idx = i

            # Check overlap with right foot active steps
            best_overlap_right = 0
            best_right_idx = -1
            for i, step in enumerate(active_right_steps):
                step_bbox = step[-1]['bbox']  # Last bbox in the current active step
                overlap = calculate_overlap(bbox, step_bbox)
                if overlap > best_overlap_right:
                    best_overlap_right = overlap
                    best_right_idx = i

            # Improved bounding box assignment logic
            if best_overlap_left > overlap_threshold and best_overlap_left > best_overlap_right:
                active_left_steps[best_left_idx].append({'bbox': bbox, 'timepoint': frame_timepoint})
                updated_left_steps[best_left_idx] = True
                left_foot_active = True
            elif best_overlap_right > overlap_threshold:
                active_right_steps[best_right_idx].append({'bbox': bbox, 'timepoint': frame_timepoint})
                updated_right_steps[best_right_idx] = True
                right_foot_active = True
            else:
                # New step detection
                if not left_foot_active:
                    # Left foot is inactive, assign to left
                    active_left_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                    updated_left_steps.append(True)
                    left_foot_active = True
                elif not right_foot_active:
                    # Right foot is inactive, assign to right
                    active_right_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                    updated_right_steps.append(True)
                    right_foot_active = True
                else:
                    # Assign based on vertical position (initial standing phase or unexpected behavior)
                    if bbox[0] < 19:  # Assume rows < 19 indicate left foot
                        active_left_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                        updated_left_steps.append(True)
                        left_foot_active = True
                    else:
                        active_right_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                        updated_right_steps.append(True)
                        right_foot_active = True
            
            # Update foot activity
            if not any(updated_left_steps):
                left_foot_active = False
            if not any(updated_right_steps):
                right_foot_active = False

        # Finalize completed steps
        steps['left'] += [step for i, step in enumerate(active_left_steps) if not updated_left_steps[i]]
        steps['right'] += [step for i, step in enumerate(active_right_steps) if not updated_right_steps[i]]

        # Keep only active steps
        active_left_steps = [step for i, step in enumerate(active_left_steps) if updated_left_steps[i]]
        active_right_steps = [step for i, step in enumerate(active_right_steps) if updated_right_steps[i]]

    return steps

def filter_short_steps(steps, min_frames=4):
    """
    Filter out steps with fewer than `min_frames` frames.
    """
    filtered_steps = {'left': [], 'right': []}
    for foot in ['left', 'right']:
        for step in steps[foot]:
            if len(step) >= min_frames:
                filtered_steps[foot].append(step)
    return filtered_steps

def generate_combined_matrices2(df):
    """
    Generate a list of combined matrices for each sample.
    Args:
        df: DataFrame containing sensor data.
    Returns:
        Dictionary of combined matrices with sample numbers as keys.
    """
    combined_matrices = {}

    # Group by sample and generate combined matrices
    for sample, rows in df.groupby('sample'):
        combined_matrix = combine_mats(rows)
        combined_matrices[sample] = combined_matrix

    return combined_matrices

def reassign_steps_by_alternation(steps):
    """
    Combine all steps and reassign left/right based on temporal alternation.
    """
    # Combine all steps with an identifier for original foot
    all_steps = [{'foot': 'left', 'step': step} for step in steps['left']]
    all_steps += [{'foot': 'right', 'step': step} for step in steps['right']]

    # Sort steps by their first timestamp
    all_steps.sort(key=lambda x: x['step'][0]['timepoint'])

    # Reassign left and right based on alternation
    reassigned_steps = {'left': [], 'right': []}
    previous_foot = None  # Start with no previous step

    for item in all_steps:
        step = item['step']
        avg_position = np.mean([bbox['bbox'][0] for bbox in step])  # Vertical position

        if previous_foot is None:
            # Initial assignment based on position
            if avg_position < 19:
                reassigned_steps['left'].append(step)
                previous_foot = 'left'
            else:
                reassigned_steps['right'].append(step)
                previous_foot = 'right'
        else:
            # Alternate feet
            if previous_foot == 'left':
                reassigned_steps['right'].append(step)
                previous_foot = 'right'
            else:
                reassigned_steps['left'].append(step)
                previous_foot = 'left'

    return reassigned_steps

def organize_steps_with_pressures(steps, combined_matrices_by_time, df):
    """
    Organize steps into a structured format, including all pressure values.
    """
    structured_steps = []

    for foot in ['left', 'right']:
        for i, step in enumerate(steps[foot]):
            step_data = {
                'foot': foot,
                'step_id': i + 1,
                'frames': [],
                'start_time': step[0]['timepoint'],
                'end_time': step[-1]['timepoint'],
            }

            for frame in step:
                bbox = frame['bbox']
                timepoint = frame['timepoint']

                # Find the sample number corresponding to the current timepoint
                rows = df[df['timepoint'] == timepoint]
                if rows.empty:
                    continue

                sample = rows['sample'].iloc[0]

                # Get the combined matrix for the sample
                combined_matrix = combined_matrices_by_time[sample]

                # Extract the sub-matrix for the bounding box
                min_row, min_col, max_row, max_col = bbox
                sub_matrix = combined_matrix[min_row:max_row + 1, min_col:max_col + 1]

                # Create frame data with pressures
                frame_data = {
                    'bbox': bbox,  # Bounding box coordinates
                    'timepoint': timepoint,  # Time of the frame
                    'pressures': sub_matrix.tolist(),  # Pressure values
                }
                step_data['frames'].append(frame_data)

            structured_steps.append(step_data)

    return structured_steps

def calculate_overbox(step):
    """
    Calculate the overbox for a step.
    """
    min_row = min(frame['bbox'][0] for frame in step['frames'])
    min_col = min(frame['bbox'][1] for frame in step['frames'])
    max_row = max(frame['bbox'][2] for frame in step['frames'])
    max_col = max(frame['bbox'][3] for frame in step['frames'])
    return min_row, min_col, max_row, max_col

def add_step_metadata_with_overbox(structured_steps, combined_matrices_by_sample, df):
    """
    Add metadata (heel strike, heel off, toe strike, toe off) using the overbox for each step.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")

    for step in structured_steps:
        # Calculate the overbox
        overbox = calculate_overbox(step)

        # Metadata initialization
        heel_strike = step['start_time']
        toe_off = step['end_time']
        heel_off = None
        toe_strike = None

        # Calculate maximum pressures for heel and toe regions
        max_heel_pressure = 0
        max_toe_pressure = 0

        for frame in step['frames']:
            timepoint = frame['timepoint']
            sample_row = df.loc[df['timepoint'] == timepoint]
            if sample_row.empty:
                continue

            sample = sample_row['sample'].values[0]

            if sample not in combined_matrices_by_sample:
                continue

            # Get the combined matrix for the sample
            matrix = combined_matrices_by_sample[sample]

            # Extract sub-matrix for the overbox
            min_row, min_col, max_row, max_col = overbox
            overbox_matrix = matrix[min_row:max_row + 1, min_col:max_col + 1]

            # Correctly divide the overbox into heel and toe regions (left and right halves)
            mid_col = overbox_matrix.shape[1] // 2
            heel_region = overbox_matrix[:, :mid_col]  # Left half (heel)
            toe_region = overbox_matrix[:, mid_col:]  # Right half (toe)

            # Update maximum pressures
            max_heel_pressure = max(max_heel_pressure, heel_region.sum())
            max_toe_pressure = max(max_toe_pressure, toe_region.sum())

        for frame in step['frames']:
            timepoint = frame['timepoint']
            sample_row = df.loc[df['timepoint'] == timepoint]
            if sample_row.empty:
                continue

            sample = sample_row['sample'].values[0]
            if sample not in combined_matrices_by_sample:
                continue

            matrix = combined_matrices_by_sample[sample]
            min_row, min_col, max_row, max_col = overbox
            overbox_matrix = matrix[min_row:max_row + 1, min_col:max_col + 1]
            mid_col = overbox_matrix.shape[1] // 2
            heel_region = overbox_matrix[:, :mid_col]
            toe_region = overbox_matrix[:, mid_col:]

            # Calculate pressures for the current frame
            heel_pressure = heel_region.sum()
            toe_pressure = toe_region.sum()

            # Detect events based on relative pressures
            if heel_off is None and heel_pressure < 0.2 * max_heel_pressure:
                heel_off = timepoint
            if toe_strike is None and toe_pressure > 0.2 * max_toe_pressure:
                toe_strike = timepoint

        # Add metadata to the step
        step['metadata'] = {
            'heel_strike': heel_strike,
            'heel_off': heel_off,
            'toe_strike': toe_strike,
            'toe_off': toe_off,
            'overbox': overbox,
        }

    return structured_steps

def analyze_gait_cycles_with_metrics(structured_steps):
    """
    Analyze gait cycles to extract metrics: stance time, swing time, double/single support times, etc.
    """
    # Sort steps by heel_strike timestamp
    structured_steps = sorted(structured_steps, key=lambda step: step['metadata']['heel_strike'])

    # Ensure alternating steps between left and right feet
    alternating_steps = []
    previous_foot = None

    for step in structured_steps:
        if previous_foot is None or step['foot'] != previous_foot:
            alternating_steps.append(step)
            previous_foot = step['foot']

    gait_cycles = []

    for i in range(len(alternating_steps) - 2):
        current_step = alternating_steps[i]
        next_step = alternating_steps[i + 1]
        next_next_step = alternating_steps[i + 2]

        # Ensure we have heel-strike and toe-off metadata
        if not current_step['metadata'] or not next_step['metadata']:
            continue

        # Determine if this forms a valid gait cycle (alternating feet)
        if current_step['foot'] == 'right' and next_step['foot'] == 'left':
            right_heel_strike = current_step['metadata']['heel_strike']
            right_toe_off = current_step['metadata']['toe_off']
            left_heel_strike = next_step['metadata']['heel_strike']
            left_toe_off = next_step['metadata']['toe_off']
            right_next_cycle_heel_strike = next_next_step['metadata']['heel_strike']

            # Gait cycle duration
            gait_cycle_duration = (right_next_cycle_heel_strike - right_heel_strike).total_seconds()

            # Stance and swing times
            right_stance_time = (right_toe_off - right_heel_strike).total_seconds()
            right_swing_time = (right_next_cycle_heel_strike - right_toe_off).total_seconds()

            left_stance_time = (right_next_cycle_heel_strike - left_heel_strike).total_seconds()
            left_swing_time = (right_heel_strike - left_toe_off).total_seconds()

            # Double and single support times
            double_support_time = (left_toe_off - right_heel_strike).total_seconds() + (right_toe_off - left_heel_strike).total_seconds()
            single_support_time = gait_cycle_duration - double_support_time

            # Cadence (steps per minute)
            cadence = 60 / gait_cycle_duration if gait_cycle_duration > 0 else 0

            # Append cycle details
            gait_cycles.append({
                'gait_cycle_id': len(gait_cycles) + 1,
                'right_leg': {
                    'heel_strike': right_heel_strike,
                    'toe_off': right_toe_off,
                    'stance_time': right_stance_time,
                    'swing_time': right_swing_time,
                },
                'left_leg': {
                    'heel_strike': left_heel_strike,
                    'toe_off': left_toe_off,
                    'stance_time': left_stance_time,
                    'swing_time': left_swing_time,
                },
                'double_support_time': double_support_time,
                'single_support_time': single_support_time,
                'cadence': cadence,
                'gait_cycle_duration': gait_cycle_duration,
            })

    return gait_cycles

def calculate_gait_cycle_distances(gait_cycles, structured_steps, distance_per_data_point=2):
    """
    Calculate average horizontal and vertical distances between feet for each gait cycle.
    
    NOTE: This calculates step width (lateral distance between left/right feet), 
    NOT stride length (forward progression). For stride length, we need forward distance
    between consecutive heel strikes of the same foot.
    
    However, for plotting purposes, we use this as an approximation of stride length
    by calculating the forward progression distance between consecutive steps.
    """
    # Map steps by time for easy lookup
    steps_by_time = {
        step['metadata']['heel_strike']: step for step in structured_steps
        if 'metadata' in step and 'heel_strike' in step['metadata']
    }

    for cycle in gait_cycles:
        # Ensure valid gait cycle metadata
        if 'right_leg' in cycle and 'left_leg' in cycle:
            right_heel_strike = cycle['right_leg']['heel_strike']
            left_heel_strike = cycle['left_leg']['heel_strike']

            if right_heel_strike in steps_by_time and left_heel_strike in steps_by_time:
                right_step = steps_by_time[right_heel_strike]
                left_step = steps_by_time[left_heel_strike]

                # Extract overbox positions
                right_overbox = right_step['metadata']['overbox']
                left_overbox = left_step['metadata']['overbox']

                # Calculate horizontal and vertical distances
                right_center_x = (right_overbox[1] + right_overbox[3]) / 2
                right_center_y = (right_overbox[0] + right_overbox[2]) / 2

                left_center_x = (left_overbox[1] + left_overbox[3]) / 2
                left_center_y = (left_overbox[0] + left_overbox[2]) / 2

                horizontal_distance = abs(right_center_x - left_center_x) * distance_per_data_point
                vertical_distance = abs(right_center_y - left_center_y) * distance_per_data_point

                # Validate: Normal stride/step length should be 30-150cm. If > 150cm, likely includes obstacle or error
                # Filter out unrealistic values that might include obstacle
                if horizontal_distance > 150:  # Unrealistic, likely includes obstacle
                    cycle['average_horizontal_distance'] = None
                else:
                    cycle['average_horizontal_distance'] = horizontal_distance
                cycle['average_vertical_distance'] = vertical_distance
            else:
                cycle['average_horizontal_distance'] = None
                cycle['average_vertical_distance'] = None
        else:
            cycle['average_horizontal_distance'] = None
            cycle['average_vertical_distance'] = None

    return gait_cycles

def extract_center_of_pressure_trace(structured_steps, combined_matrices_by_sample, df):
    cop_traces = []

    for step in structured_steps:
        foot = step['foot']
        step_id = step['step_id']
        overbox = step['metadata']['overbox']

        cop_x = []
        cop_y = []

        for frame in step['frames']:
            timepoint = frame['timepoint']
            sample_row = df.loc[df['timepoint'] == timepoint]
            if sample_row.empty:
                continue
            sample = sample_row['sample'].values[0]
            if sample not in combined_matrices_by_sample:
                continue

            matrix = combined_matrices_by_sample[sample]
            min_row, min_col, max_row, max_col = overbox
            sub_matrix = matrix[min_row:max_row + 1, min_col:max_col + 1]

            pressure_sum = sub_matrix.sum()
            if pressure_sum > 0:
                y_coords, x_coords = np.meshgrid(
                    np.arange(min_row, max_row + 1),
                    np.arange(min_col, max_col + 1),
                    indexing="ij"
                )
                cop_x.append((sub_matrix * x_coords).sum() / pressure_sum)
                cop_y.append((sub_matrix * y_coords).sum() / pressure_sum)

        cop_traces.append({
            "foot": foot,
            "step_id": step_id,
            "x": cop_x,
            "y": cop_y
        })

    return cop_traces

