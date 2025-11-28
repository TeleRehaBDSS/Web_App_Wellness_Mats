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
    for _, row in rows.iterrows():
        try:
            mat_index = int(row['mat'])
            if not 0 <= mat_index < 6:  # Updated range
                print(f"Invalid mat_index: {mat_index}")
                continue
            sensor_data = parse_and_rotate_sensor_data(row['sensors'])
            if sensor_data.shape != (48, 48):
                print(f"Unexpected sensor_data shape: {sensor_data.shape}")
                continue
            combined[:, mat_index * 48:(mat_index + 1) * 48] = sensor_data
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    if combined.sum() == 0:
        print("Warning: Combined matrix is empty.")
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

# Track steps over frames
def track_steps(bboxes_by_frame, timepoints, overlap_threshold=0.5):
    steps = []
    active_steps = []
    for frame_idx, bboxes in enumerate(bboxes_by_frame):
        updated_steps = [False] * len(active_steps)
        frame_timepoint = timepoints[frame_idx]
        for bbox in bboxes:
            matched = False
            for i, step in enumerate(active_steps):
                step_bbox = step[-1]['bbox']
                overlap = calculate_overlap(step_bbox, bbox)
                if overlap > overlap_threshold:
                    step.append({'bbox': bbox, 'timepoint': frame_timepoint})
                    updated_steps[i] = True
                    matched = True
                    break
            if not matched:
                active_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                updated_steps.append(True)
        steps += [step for i, step in enumerate(active_steps) if not updated_steps[i]]
        active_steps = [step for i, step in enumerate(active_steps) if updated_steps[i]]
    steps += active_steps
    return steps

def calculate_overlap(bbox1, bbox2):
    min_row1, min_col1, max_row1, max_col1 = bbox1
    min_row2, min_col2, max_row2, max_col2 = bbox2
    overlap_row = max(0, min(max_row1, max_row2) - max(min_row1, min_row2))
    overlap_col = max(0, min(max_col1, max_col2) - max(min_col1, min_col2))
    overlap_area = overlap_row * overlap_col
    bbox1_area = (max_row1 - min_row1) * (max_col1 - min_col1)
    bbox2_area = (max_row2 - min_row2) * (max_col2 - min_col2)
    return overlap_area / max(bbox1_area, bbox2_area) if max(bbox1_area, bbox2_area) > 0 else 0

# Calculate statistics
def calculate_statistics(steps):
    num_steps = len(steps)
    durations = [(step[-1]['timepoint'] - step[0]['timepoint']).total_seconds() for step in steps]
    return num_steps, durations

# Update Function for Animation
def update(frame):

    # Setup Figure for Animation
    fig, ax = plt.subplots(figsize=(15, 6))
    cax = ax.matshow(np.zeros((48, 48 * 6)), cmap='hot', vmin=0, vmax=100)
    ax.set_title("Pressure Mats with Step Detection")

    bboxes_by_frame = []
    timepoints = []

    rows = df[df['sample'] == frame]
    timepoint = rows.iloc[0]['timepoint']
    #print("Rows being combined:")
    #print(rows)

    combined_mat = combine_mats(rows)
    bboxes = detect_clusters(combined_mat, threshold=5)
    bboxes_by_frame.append(bboxes)
    timepoints.append(timepoint)
    cax.set_data(combined_mat)
    ax.clear()
    ax.matshow(combined_mat, cmap='hot', vmin=0, vmax=200)
    ax.set_title(f"Timepoint: {timepoint}")
    for bbox in bboxes:
        min_row, min_col, max_row, max_col = bbox
        rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor='cyan', linewidth=2, fill=False)
        ax.add_patch(rect)
    return cax,


# Track steps over frames as groups of successive bounding boxes
def track_steps_by_grouping(bboxes_by_frame, timepoints, overlap_threshold=0.5):
    """
    Group bounding boxes into steps by tracking them across successive frames.
    Each step is a collection of overlapping or close bboxes across frames.
    """
    steps = []  # List to hold all detected steps
    active_steps = []  # Steps currently being tracked

    for frame_idx, bboxes in enumerate(bboxes_by_frame):
        frame_timepoint = timepoints[frame_idx]
        updated_steps = [False] * len(active_steps)

        for bbox in bboxes:
            matched = False
            for i, step in enumerate(active_steps):
                step_bbox = step[-1]['bbox']  # Last bbox in the current active step
                overlap = calculate_overlap(step_bbox, bbox)
                print('overlap = ', overlap)
                if overlap > overlap_threshold:
                    # Add bbox to the active step
                    step.append({'bbox': bbox, 'timepoint': frame_timepoint})
                    updated_steps[i] = True
                    matched = True
                    break

            if not matched:
                # Start a new step
                print('start a new step')
                active_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                updated_steps.append(True)

        # Add completed steps to final list
        steps += [step for i, step in enumerate(active_steps) if not updated_steps[i]]

        # Keep only active steps
        active_steps = [step for i, step in enumerate(active_steps) if updated_steps[i]]

    # Add remaining active steps to final list
    steps += active_steps
    return steps


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


import matplotlib.pyplot as plt

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
        print('bboxes = ', bboxes)

        updated_left_steps = [False] * len(active_left_steps)
        updated_right_steps = [False] * len(active_right_steps)

        for bbox in bboxes:
            print('I am checking bbox = ', bbox)
            # Check overlap with left foot active steps
            best_overlap_left = 0
            best_left_idx = -1
            for i, step in enumerate(active_left_steps):
                step_bbox = step[-1]['bbox']  # Last bbox in the current active step
                overlap = calculate_overlap(bbox, step_bbox)
                print(f'overlap {i} with the left', overlap)
                if overlap > best_overlap_left:
                    best_overlap_left = overlap
                    best_left_idx = i

            # Check overlap with right foot active steps
            best_overlap_right = 0
            best_right_idx = -1
            for i, step in enumerate(active_right_steps):
                step_bbox = step[-1]['bbox']  # Last bbox in the current active step
                overlap = calculate_overlap(bbox, step_bbox)
                print(f'overlap {i} with the right', overlap)
                if overlap > best_overlap_right:
                    best_overlap_right = overlap
                    best_right_idx = i

            # Improved bounding box assignment logic
            if best_overlap_left > overlap_threshold and best_overlap_left > best_overlap_right:
                active_left_steps[best_left_idx].append({'bbox': bbox, 'timepoint': frame_timepoint})
                updated_left_steps[best_left_idx] = True
                print('assign to left')
                left_foot_active = True
            elif best_overlap_right > overlap_threshold:
                active_right_steps[best_right_idx].append({'bbox': bbox, 'timepoint': frame_timepoint})
                updated_right_steps[best_right_idx] = True
                print('assign to right')
                right_foot_active = True
            else:
                # New step detection
                if not left_foot_active:
                    # Left foot is inactive, assign to left
                    active_left_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                    updated_left_steps.append(True)
                    left_foot_active = True
                    print('new step for left (reappeared after inactivity)')
                elif not right_foot_active:
                    # Right foot is inactive, assign to right
                    active_right_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                    updated_right_steps.append(True)
                    right_foot_active = True
                    print('new step for right (reappeared after inactivity)')
                else:
                    # Assign based on vertical position (initial standing phase or unexpected behavior)
                    if bbox[0] < 19:  # Assume rows < 19 indicate left foot
                        active_left_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                        updated_left_steps.append(True)
                        print('new step for left (based on position)')
                        left_foot_active = True
                    else:
                        active_right_steps.append([{'bbox': bbox, 'timepoint': frame_timepoint}])
                        updated_right_steps.append(True)
                        print('new step for right (based on position)')
                        right_foot_active = True
                # Debug plot for new step
                #plt.figure(figsize=(8, 8))
                #plt.title(f"New Step Detected (Frame {frame_idx})")
                #plt.imshow(np.zeros((48, 48 * 6)), cmap='hot', vmin=0, vmax=100)
                #min_row, min_col, max_row, max_col = bbox
                #plt.gca().add_patch(plt.Rectangle(
                #    (min_col, min_row),  # Bottom-left corner
                #    max_col - min_col,   # Width
                #    max_row - min_row,   # Height
                #    edgecolor='cyan' if right_foot_active else 'magenta',
                #    linewidth=2,
                #    fill=False
                #))
                #plt.show()
                #input("Press Enter to continue...")

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

def print_step_statistics_with_pressures(steps, avg_pressures):
    """
    Print the number of frames and average pressures for each detected step.
    """
    print("\nLeft Foot Steps:")
    for i, step in enumerate(steps['left']):
        num_frames = len(step)
        avg_pressure = avg_pressures['left'][i]
        print(f"Step {i + 1}: {num_frames} frames, Average Pressure: {avg_pressure:.2f}")

    print("\nRight Foot Steps:")
    for i, step in enumerate(steps['right']):
        num_frames = len(step)
        avg_pressure = avg_pressures['right'][i]
        print(f"Step {i + 1}: {num_frames} frames, Average Pressure: {avg_pressure:.2f}")


# After the track_steps_separately function is called
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



def calculate_average_pressure_with_samples(steps, df):
    """
    Calculate the average pressure for each step using sample numbers for consistency.
    Args:
        steps: Dictionary containing left and right steps.
        df: DataFrame containing sensor data.
    Returns:
        Dictionary with average pressure for each step.
    """
    avg_pressures = {'left': [], 'right': []}

    for foot in ['left', 'right']:
        for step in steps[foot]:
            total_pressure = 0
            total_cells = 0

            for frame in step:
                timepoint = frame['timepoint']
                bbox = frame['bbox']

                # Map timepoint to sample number
                sample_numbers = df[df['timepoint'] == timepoint]['sample'].unique()
                if len(sample_numbers) == 0:
                    print(f"No sample found for timepoint: {timepoint}")
                    continue
                sample = sample_numbers[0]

                # Fetch rows for the sample
                rows = df[df['sample'] == sample]
                if rows.empty:
                    print(f"No rows found for sample: {sample}")
                    continue

                # Combine sensor data for the current sample
                combined_matrix = combine_mats(rows)

                # Extract the sub-matrix for the bounding box
                min_row, min_col, max_row, max_col = bbox
                sub_matrix = combined_matrix[min_row:max_row + 1, min_col:max_col + 1]

                # Sum pressures and count cells
                total_pressure += sub_matrix.sum()
                total_cells += sub_matrix.size

            # Calculate average pressure for the step
            avg_pressure = total_pressure / total_cells if total_cells > 0 else 0
            avg_pressures[foot].append(avg_pressure)

    return avg_pressures


def visualize_steps(steps, df):
    """
    Visualize each step with the heatmap and bounding boxes.
    Uses sample numbers to ensure consistency with the update function.
    """
    for foot in ['left', 'right']:
        for i, step in enumerate(steps[foot]):
            print(f"Visualizing {foot.capitalize()} Step {i + 1}")
            fig, ax = plt.subplots(figsize=(10, 6))

            for frame in step:
                timepoint = frame['timepoint']
                bbox = frame['bbox']

                # Map timepoint to sample number
                sample_numbers = df[df['timepoint'] == timepoint]['sample'].unique()
                if len(sample_numbers) == 0:
                    print(f"No sample found for timepoint: {timepoint}")
                    continue
                sample = sample_numbers[0]

                # Fetch rows for the sample
                rows = df[df['sample'] == sample]
                if rows.empty:
                    print(f"No rows found for sample: {sample}")
                    continue

                # Construct the combined matrix
                combined_matrix = combine_mats(rows)
                print(f"Combined Matrix Sum for Sample {sample}: {combined_matrix.sum()}")

                # Display the heatmap
                ax.matshow(combined_matrix, cmap='hot', vmin=0, vmax=100)

                # Draw the bounding box
                min_row, min_col, max_row, max_col = bbox
                rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                     edgecolor='cyan', linewidth=2, fill=False)
                ax.add_patch(rect)

            ax.set_title(f"{foot.capitalize()} Step {i + 1}")
            plt.show()





def generate_combined_matrices(df, timepoints):
    """
    Generate a list of combined matrices for each timepoint.
    Args:
        df: DataFrame containing sensor data.
        timepoints: List of unique timepoints.
    Returns:
        List of combined matrices for each timepoint.
    """
    combined_matrices = []
    
    for timepoint in timepoints:
        # Select rows corresponding to the current timepoint
        rows = df[df['sample'] == timepoint]
        combined_matrix = combine_mats(rows)
        combined_matrices.append(combined_matrix)
    
    return combined_matrices

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
    Args:
        steps: Dictionary containing left and right steps.
        combined_matrices_by_time: List of combined matrices corresponding to sample numbers.
        df: DataFrame containing the sensor data.
    Returns:
        List of structured steps with detailed information, including pressure values.
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
                    print(f"Timepoint {timepoint} not found in DataFrame.")
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
    Calculate the overbox for a step, which is the smallest bounding box that contains all individual boxes.
    Args:
        step: List of frames with bounding boxes.
    Returns:
        Tuple representing the overbox (min_row, min_col, max_row, max_col).
    """
    min_row = min(frame['bbox'][0] for frame in step['frames'])
    min_col = min(frame['bbox'][1] for frame in step['frames'])
    max_row = max(frame['bbox'][2] for frame in step['frames'])
    max_col = max(frame['bbox'][3] for frame in step['frames'])
    return min_row, min_col, max_row, max_col


def add_step_metadata_with_overbox(structured_steps, combined_matrices_by_sample, df):
    """
    Add metadata (heel strike, heel off, toe strike, toe off) using the overbox for each step.
    Args:
        structured_steps: List of steps with bounding boxes and timepoints.
        combined_matrices_by_sample: Dictionary of combined matrices indexed by sample.
        df: DataFrame containing the sensor data, including sample numbers.
    Returns:
        List of steps with added metadata.
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
                print(f"Timepoint {timepoint} not found in DataFrame.")
                continue

            sample = sample_row['sample'].values[0]

            if sample not in combined_matrices_by_sample:
                print(f"Sample {sample} not found in combined_matrices_by_sample.")
                continue

            # Get the combined matrix for the sample
            matrix = combined_matrices_by_sample[sample]

            # Extract sub-matrix for the overbox
            min_row, min_col, max_row, max_col = overbox
            overbox_matrix = matrix[min_row:max_row + 1, min_col:max_col + 1]
            print(f"Overbox matrix for sample {sample}:\n", overbox_matrix)

            # Correctly divide the overbox into heel and toe regions (left and right halves)
            mid_col = overbox_matrix.shape[1] // 2
            heel_region = overbox_matrix[:, :mid_col]  # Left half (heel)
            toe_region = overbox_matrix[:, mid_col:]  # Right half (toe)
            print("Heel region:\n", heel_region)
            print("Toe region:\n", toe_region)

            # Update maximum pressures
            max_heel_pressure = max(max_heel_pressure, heel_region.sum())
            max_toe_pressure = max(max_toe_pressure, toe_region.sum())

        print(f"Max heel pressure: {max_heel_pressure}, Max toe pressure: {max_toe_pressure}")

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
            print(f"Frame {frame}: Heel pressure = {heel_pressure}, Toe pressure = {toe_pressure}")

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




import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_step_overbox(structured_steps, combined_matrices_by_sample, df):
    """
    Create animations for each step visualizing the overbox and its pressure evolution.
    Args:
        structured_steps: List of steps with bounding boxes and timepoints.
        combined_matrices_by_sample: Dictionary of combined matrices indexed by sample.
        df: DataFrame containing the sensor data, including sample numbers.
    """
    for step in structured_steps:
        # Calculate the overbox
        overbox = calculate_overbox(step)
        min_row, min_col, max_row, max_col = overbox

        fig, ax = plt.subplots(figsize=(10, 6))
        print(f"Animating Step {step['step_id']} ({step['foot']} foot)")

        def update(frame):
            ax.clear()
            timepoint = frame['timepoint']
            
            # Get the corresponding sample number for the current timepoint
            sample_row = df.loc[df['timepoint'] == timepoint]
            if sample_row.empty:
                print(f"Timepoint {timepoint} not found in DataFrame.")
                return
            sample = sample_row['sample'].values[0]

            if sample not in combined_matrices_by_sample:
                print(f"Sample {sample} not found in combined_matrices_by_sample.")
                return
            
            # Get the combined matrix for the current sample
            matrix = combined_matrices_by_sample[sample]

            # Plot the full heatmap
            ax.matshow(matrix, cmap='hot', vmin=0, vmax=100)
            ax.set_title(f"Step {step['step_id']} ({step['foot']} foot), Frame Time: {timepoint}")

            # Highlight the overbox
            rect = plt.Rectangle(
                (min_col, min_row),
                max_col - min_col,
                max_row - min_row,
                edgecolor='cyan',
                linewidth=2,
                fill=False
            )
            ax.add_patch(rect)

            # Extract and plot the overbox region separately
            overbox_matrix = matrix[min_row:max_row + 1, min_col:max_col + 1]
            ax_inset = fig.add_axes([0.7, 0.7, 0.2, 0.2])
            ax_inset.matshow(overbox_matrix, cmap='hot', vmin=0, vmax=100)
            ax_inset.set_title("Overbox")

        ani = animation.FuncAnimation(
            fig, update, frames=step['frames'], repeat=False
        )
        plt.show()

#animate_step_overbox(structured_steps, combined_matrix_by_time, df)



def analyze_gait_cycles(gait_cycles):
    """
    Perform gait analysis using gait cycles.

    Args:
        gait_cycles: List of gait cycles containing metadata for left and right steps.

    Returns:
        Dictionary with gait analysis metrics.
    """
    gait_analysis = {
        'number_of_steps': 0,
        'cadence': 0,
        'single_support_times': [],
        'double_support_times': [],
        'swing_times': [],
    }

    total_time = 0
    total_steps = 0

    for i in range(len(gait_cycles)):
        cycle = gait_cycles[i]

        left_step = cycle['left_step']
        right_step = cycle['right_step']

        # Increment total steps
        if left_step:
            total_steps += 1
        if right_step:
            total_steps += 1

        # Calculate single support time
        if left_step and right_step:
            left_support = (
                right_step['metadata']['heel_strike'] - left_step['metadata']['toe_off']
            ).total_seconds() if right_step['metadata']['heel_strike'] and left_step['metadata']['toe_off'] else None

            right_support = (
                left_step['metadata']['heel_strike'] - right_step['metadata']['toe_off']
            ).total_seconds() if left_step['metadata']['heel_strike'] and right_step['metadata']['toe_off'] else None

            if left_support:
                gait_analysis['single_support_times'].append(left_support)
            if right_support:
                gait_analysis['single_support_times'].append(right_support)

        # Calculate double support times
        if i < len(gait_cycles) - 1:  # Ensure we have the next cycle
            next_cycle = gait_cycles[i + 1]
            next_left_step = next_cycle['left_step']
            next_right_step = next_cycle['right_step']

            # Double support 1: Right Toe Off to Left Heel Strike
            if right_step and next_left_step:
                double_support_1 = (
                    next_left_step['metadata']['toe_off'] - right_step['metadata']['heel_strike']
                ).total_seconds() if next_left_step['metadata']['toe_off'] and right_step['metadata']['heel_strike'] else None

                if double_support_1:
                    gait_analysis['double_support_times'].append(double_support_1)

            # Double support 2: Left Toe Off to Right Heel Strike
            if left_step and next_right_step:
                double_support_2 = (
                    next_right_step['metadata']['toe_off'] - left_step['metadata']['heel_strike']
                ).total_seconds() if next_right_step['metadata']['toe_off'] and left_step['metadata']['heel_strike'] else None

                if double_support_2:
                    gait_analysis['double_support_times'].append(double_support_2)

        # Calculate swing times
        if left_step and right_step:
            left_swing = (
                left_step['metadata']['toe_off'] - left_step['metadata']['heel_strike']
            ).total_seconds() if left_step['metadata']['toe_off'] and left_step['metadata']['heel_strike'] else None

            right_swing = (
                right_step['metadata']['toe_off'] - right_step['metadata']['heel_strike']
            ).total_seconds() if right_step['metadata']['toe_off'] and right_step['metadata']['heel_strike'] else None

            if left_swing:
                gait_analysis['swing_times'].append(left_swing)
            if right_swing:
                gait_analysis['swing_times'].append(right_swing)

        # Update total time (for cadence calculation)
        if left_step:
            total_time += (left_step['metadata']['toe_off'] - left_step['metadata']['heel_strike']).total_seconds() if left_step['metadata']['toe_off'] and left_step['metadata']['heel_strike'] else 0
        if right_step:
            total_time += (right_step['metadata']['toe_off'] - right_step['metadata']['heel_strike']).total_seconds() if right_step['metadata']['toe_off'] and right_step['metadata']['heel_strike'] else 0

    # Calculate metrics
    gait_analysis['number_of_steps'] = total_steps
    gait_analysis['cadence'] = (total_steps / total_time) * 60 if total_time > 0 else 0  # Steps per minute

    return gait_analysis



def extract_gait_cycles(structured_steps):
    """
    Extract gait cycles from structured steps metadata.
    """
    gait_cycles = []
    left_steps = [step for step in structured_steps if step['foot'] == 'left']
    right_steps = [step for step in structured_steps if step['foot'] == 'right']

    for i in range(min(len(left_steps), len(right_steps)) - 1):
        cycle = {
            'left_step': left_steps[i],
            'right_step': right_steps[i],
            'next_left_step': left_steps[i + 1],
            'next_right_step': right_steps[i + 1],
        }
        gait_cycles.append(cycle)
    return gait_cycles


def analyze_gait_cycles_with_metrics(structured_steps):
    """
    Analyze gait cycles to extract metrics: stance time, swing time, double/single support times, etc.
    Args:
        structured_steps: List of steps with metadata (heel-strike, toe-off, etc.).
    Returns:
        List of gait cycles with detailed metrics.
    """
    # Sort steps by heel_strike timestamp
    print("structured_steps = ", structured_steps)
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
            print('right_next_cycle_heel_strike = ', right_next_cycle_heel_strike)

            # Gait cycle duration
            gait_cycle_duration = (right_next_cycle_heel_strike - right_heel_strike).total_seconds()

            # Stance and swing times
            right_stance_time = (right_toe_off - right_heel_strike).total_seconds()
            right_swing_time = (right_next_cycle_heel_strike - right_toe_off).total_seconds()

            left_stance_time = (right_next_cycle_heel_strike - left_heel_strike).total_seconds()
            left_swing_time = (right_heel_strike - left_toe_off).total_seconds()

            # Double and single support times
            #double_support_time = (left_heel_strike - right_toe_off).total_seconds()
            double_support_time = (left_toe_off - right_heel_strike).total_seconds() + (right_toe_off - left_heel_strike).total_seconds() ####check this 20250104
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



import matplotlib.pyplot as plt

def visualize_gait_analysis(gait_analysis):
    plt.figure(figsize=(10, 6))

    # Gait Cycle Times
    plt.subplot(2, 2, 1)
    plt.plot(gait_analysis['gait_cycles'], label='Gait Cycle Time')
    plt.title('Gait Cycle Time')
    plt.xlabel('Gait Cycle Index')
    plt.ylabel('Time (s)')

    # Single Support Times
    plt.subplot(2, 2, 2)
    plt.plot(gait_analysis['single_support_times'], label='Single Support Time', color='orange')
    plt.title('Single Support Time')
    plt.xlabel('Gait Cycle Index')
    plt.ylabel('Time (s)')

    # Double Support Times
    plt.subplot(2, 2, 3)
    plt.plot(gait_analysis['double_support_times'], label='Double Support Time', color='green')
    plt.title('Double Support Time')
    plt.xlabel('Gait Cycle Index')
    plt.ylabel('Time (s)')

    # Swing Times
    plt.subplot(2, 2, 4)
    plt.plot(gait_analysis['swing_times'], label='Swing Time', color='red')
    plt.title('Swing Time')
    plt.xlabel('Gait Cycle Index')
    plt.ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

#gc = extract_gait_cycles(structured_steps)
#ga = analyze_gait_cycles(gc)
#print(ga)
#visualize_gait_analysis(ga)

import matplotlib.dates as mdates


def plot_gait_cycles(gait_cycles):
    """
    Plot each gait cycle with annotations about timings.
    Args:
        gait_cycles: List of gait cycles with detailed metrics.
    """
    for cycle in gait_cycles:
        # Extract data for the current cycle
        gait_cycle_id = cycle['gait_cycle_id']
        right_leg = cycle['right_leg']
        left_leg = cycle['left_leg']
        double_support_time = cycle['double_support_time']
        single_support_time = cycle['single_support_time']
        gait_cycle_duration = cycle['gait_cycle_duration']

        # Create a timeline for events
        events = {
            'Right Heel Strike': right_leg['heel_strike'],
            'Right Toe Off': right_leg['toe_off'],
            'Left Heel Strike': left_leg['heel_strike'],
            'Left Toe Off': left_leg['toe_off'],
        }

        # Convert timestamps to matplotlib date format for plotting
        event_times = list(events.values())
        event_labels = list(events.keys())
        event_times_mpl = mdates.date2num(event_times)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(event_times_mpl, [0] * len(event_times), 'o-', label='Gait Events')

        # Add annotations for events
        for i, (time, label) in enumerate(zip(event_times_mpl, event_labels)):
            ax.text(time, 0.1, label, rotation=45, fontsize=10, ha='right')

        # Add lines and annotations for double and single support times
        ax.plot([event_times_mpl[1], event_times_mpl[2]], [0, 0], 'r-', label='Double Support Time')
        ax.plot([event_times_mpl[0], event_times_mpl[1]], [0, 0], 'g-', label='Single Support Time (Right)')
        ax.plot([event_times_mpl[2], event_times_mpl[3]], [0, 0], 'b-', label='Single Support Time (Left)')

        # Annotate timing details
        ax.annotate(f"Double Support: {double_support_time:.2f}s",
                    xy=((event_times_mpl[1] + event_times_mpl[2]) / 2, 0),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', color='red', fontsize=10)

        ax.annotate(f"Single Support (Right): {right_leg['stance_time']:.2f}s",
                    xy=((event_times_mpl[0] + event_times_mpl[1]) / 2, 0),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', color='green', fontsize=10)

        ax.annotate(f"Single Support (Left): {left_leg['stance_time']:.2f}s",
                    xy=((event_times_mpl[2] + event_times_mpl[3]) / 2, 0),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', color='blue', fontsize=10)

        # Formatting the plot
        ax.set_title(f"Gait Cycle {gait_cycle_id}")
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        # Show plot
        plt.tight_layout()
        plt.show()

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


def plot_center_of_pressure(structured_steps, combined_matrices_by_sample, df):
    """
    Plot the trace of the center of pressure (CoP) for each step.
    Args:
        structured_steps: List of structured steps with bounding boxes, timepoints, and metadata.
        combined_matrices_by_sample: Dictionary of combined matrices indexed by sample.
        df: DataFrame containing the sensor data, including sample numbers.
    """
    for step in structured_steps:
        # Extract metadata
        foot = step['foot']
        step_id = step['step_id']
        overbox = step['metadata']['overbox']

        cop_x = []
        cop_y = []

        for frame in step['frames']:
            timepoint = frame['timepoint']
            
            # Map the timepoint to the sample
            sample_row = df.loc[df['timepoint'] == timepoint]
            if sample_row.empty:
                print(f"Timepoint {timepoint} not found in DataFrame.")
                continue
            
            sample = sample_row['sample'].values[0]
            if sample not in combined_matrices_by_sample:
                print(f"Sample {sample} not found in combined_matrices_by_sample.")
                continue

            # Get the matrix for the sample
            matrix = combined_matrices_by_sample[sample]
            
            # Extract the pressure sub-matrix for the overbox
            min_row, min_col, max_row, max_col = overbox
            sub_matrix = matrix[min_row:max_row + 1, min_col:max_col + 1]
            
            # Get coordinates of the pressure points
            pressure_sum = sub_matrix.sum()
            if pressure_sum > 0:
                rows, cols = sub_matrix.shape
                y_coords, x_coords = np.meshgrid(
                    np.arange(min_row, max_row + 1), np.arange(min_col, max_col + 1), indexing="ij"
                )
                cop_x.append((sub_matrix * x_coords).sum() / pressure_sum)
                cop_y.append((sub_matrix * y_coords).sum() / pressure_sum)

        # Plot the trace of CoP
        plt.figure(figsize=(8, 8))
        plt.plot(cop_x, cop_y, marker='o', label=f"{foot.capitalize()} Step {step_id}")
        plt.title(f"Center of Pressure Trace for {foot.capitalize()} Step {step_id}")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.gca().invert_yaxis()  # Invert y-axis for top-down view
        plt.grid(True)
        plt.legend()
        plt.show()



def calculate_gait_cycle_distances(gait_cycles, structured_steps, distance_per_data_point=2):
    """
    Calculate average horizontal and vertical distances between feet for each gait cycle.

    Args:
        gait_cycles: List of gait cycles with detailed metrics for left and right legs.
        structured_steps: List of steps with bounding boxes, timepoints, and metadata.
        distance_per_data_point: The distance represented by a single data point (in cm).

    Returns:
        Updated gait_cycles with average distances added for each cycle.
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

                # Add distances to the cycle
                cycle['average_horizontal_distance'] = horizontal_distance
                cycle['average_vertical_distance'] = vertical_distance
            else:
                # Missing valid metadata for one or both legs
                cycle['average_horizontal_distance'] = None
                cycle['average_vertical_distance'] = None
        else:
            # Missing valid leg data
            cycle['average_horizontal_distance'] = None
            cycle['average_vertical_distance'] = None

    return gait_cycles

