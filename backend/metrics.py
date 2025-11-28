import os
import pandas as pd
import numpy as np
from backend.analyze_data_functions_MAT import extract_center_of_pressure_trace
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
    generate_combined_matrices2
)

def calculate_metrics(file_path):
    """
    Calculate metrics for FGA items using pressure mat data.

    Args:
        file_path (str): Path to the uploaded CSV file.

    Returns:
        dict: Summary of calculated metrics.
    """
    try:
        # Load CSV file
        print(file_path)
        df = pd.read_csv(file_path)
        
        df['timepoint'] = pd.to_datetime(df['timepoint'])

        # Generate combined matrices for each sample
        combined_matrices_by_sample = combine_mats(df)

        # Detect clusters and track steps
        unique_samples = df['sample'].unique()
        bboxes_by_frame = []
        timepoints = []

        for sample in unique_samples:
            rows = df[df['sample'] == sample]
            combined_mat = combine_mats(rows)
            bboxes = detect_clusters(combined_mat)
            bboxes_by_frame.append(bboxes)
            timepoints.append(rows.iloc[0]['timepoint'])

        steps = track_steps_separately(bboxes_by_frame, timepoints, overlap_threshold=0.05)
        combined_matrix_by_time = generate_combined_matrices2(df)

        # Filter and reassign steps
        steps = filter_short_steps(steps, min_frames=4)
        reassigned_steps = reassign_steps_by_alternation(steps)

        # Organize steps into structured format with pressures
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
        cop_traces = extract_center_of_pressure_trace(structured_steps_with_metadata, combined_matrix_by_time, df)
        # Summarize results
        gait_analysis_summary = {
            "cop_traces": cop_traces,
            "gait_cycles": gait_cycles_with_distances,
            "number_of_steps": len(structured_steps_with_metadata),
            "average_cadence": np.mean([cycle["cadence"] for cycle in gait_cycles_with_distances]),
            "average_double_support_time": np.mean([cycle["double_support_time"] for cycle in gait_cycles_with_distances]),
            "average_single_support_time": np.mean([cycle["single_support_time"] for cycle in gait_cycles_with_distances]),
            "average_horizontal_distance": np.mean([
                cycle["average_horizontal_distance"] for cycle in gait_cycles_with_distances if cycle["average_horizontal_distance"] is not None
            ]),
            "average_vertical_distance": np.mean([
                cycle["average_vertical_distance"] for cycle in gait_cycles_with_distances if cycle["average_vertical_distance"] is not None
            ]),
        }
        

        return gait_analysis_summary

    except Exception as e:
        return {"error": f"An error occurred while processing the file: {str(e)}"}

def grade_gait_level_surface(metrics):
    """
    Grades the Gait Level Surface exercise based on calculated metrics.

    Args:
        metrics (dict): The calculated gait metrics from the backend.

    Returns:
        int: Grade (0 to 3).
        str: Explanation of the grade.
    """
    # Extract metrics
    average_cadence = metrics.get("average_cadence", 0)
    average_horizontal_distance = metrics.get("average_horizontal_distance", 0)
    number_of_steps = metrics.get("number_of_steps", 0)

    # Approximate time to walk 6m
    stride_length = average_horizontal_distance / number_of_steps if number_of_steps > 0 else 0
    time_to_walk = 6 / (average_cadence / 60) if average_cadence > 0 else float("inf")

    # Grading logic
    if time_to_walk < 5.5 and stride_length >= 30 and average_horizontal_distance <= 15.24:
        grade = 3
        explanation = (
            "Normal performance: Walks 6m in less than 5.5 seconds, no assistive device, "
            "good speed, normal gait pattern, minimal deviation."
        )
    elif 5.5 <= time_to_walk < 7 and stride_length >= 25 and average_horizontal_distance <= 25.4:
        grade = 2
        explanation = (
            "Mild impairment: Walks 6m in less than 7 seconds but greater than 5.5 seconds, "
            "slightly slower speed, mild gait deviations, or mild imbalance."
        )
    elif time_to_walk >= 7 or stride_length < 25 or average_horizontal_distance <= 38.1:
        grade = 1
        explanation = (
            "Moderate impairment: Walks 6m with slow speed, abnormal gait pattern, evidence of imbalance, "
            "or larger deviations."
        )
    else:
        grade = 0
        explanation = (
            "Severe impairment: Cannot walk 6m without assistance, severe gait deviations or imbalance, "
            "or deviates significantly."
        )
    
    return grade, explanation

def exercise02(metrics):
    """
    Grade the "Change in Gait Speed" exercise based on extracted metrics.
    
    Args:
        metrics (dict): The dictionary containing gait analysis results.
        
    Returns:
        tuple: A grade (int) and an explanation (str).
    """
    try:
        # Extract relevant metrics
        average_cadence = metrics.get("average_cadence", None)
        gait_cycles = metrics.get("gait_cycles", [])
        number_of_steps = metrics.get("number_of_steps", 0)

        # Calculate deviation in cadence across gait cycles
        cadences = [cycle.get("cadence", None) for cycle in gait_cycles if cycle.get("cadence") is not None]
        if cadences:
            cadence_mean = sum(cadences) / len(cadences)
            cadence_deviation = max(abs(c - cadence_mean) for c in cadences)
        else:
            cadence_mean = None
            cadence_deviation = None

        # Set grading thresholds
        explanation = []
        grade = 0  # Default grade
        if average_cadence is None or number_of_steps == 0 or not gait_cycles:
            return grade, "Insufficient data to grade the exercise."

        # Grade based on cadence and deviations
        if cadence_deviation is not None:
            if cadence_deviation <= 5:
                grade = 3
                explanation.append("Cadence was highly consistent across all speed transitions.")
            elif 5 < cadence_deviation <= 10:
                grade = 2
                explanation.append("Cadence was moderately consistent across speed transitions.")
            elif cadence_deviation > 10:
                grade = 1
                explanation.append("Cadence showed significant inconsistency across speed transitions.")

        # Check average cadence against expected thresholds
        if average_cadence < 40:
            explanation.append("Average cadence was too low, indicating poor performance in maintaining speed.")
        elif average_cadence >= 40:
            explanation.append("Average cadence was within an acceptable range.")

        # Check the number of steps
        if number_of_steps < 10:
            explanation.append("Too few steps were detected, which may indicate incomplete data or short walking distance.")
        elif number_of_steps >= 10:
            explanation.append("Sufficient steps were detected for grading.")

        return grade, " ".join(explanation)

    except Exception as e:
        return 0, f"An error occurred while grading the exercise: {str(e)}"


def exercise03(metrics, manual_input):
    """
    Grades the "Gait with Horizontal Head Turns" exercise based on metrics and manual input.

    Parameters:
    - metrics (dict): Automatic metrics including gait speed, deviation, and stability.
    - manual_input (str): Physio input on the smoothness of head turns:
        - "Smoothly"
        - "Mild difficulty"
        - "Moderate difficulty"
        - "Severe difficulty or unable"

    Returns:
    - grade (int): Grade (0-3) based on performance.
    - explanation (str): Explanation of the grade.
    """

    # Extract metrics
    avg_gait_speed = metrics.get("average_gait_speed", 0)  # Average speed during head turns
    deviation = metrics.get("average_horizontal_distance", 0)  # Deviation from a straight line
    stability = metrics.get("stability_score", 0)  # Hypothetical stability score

    # Grade thresholds for automatic metrics
    speed_thresholds = {
        "normal": 1.2,  # m/s
        "mild": 1.0,  # m/s
        "moderate": 0.8,  # m/s
    }

    deviation_thresholds = {
        "normal": 0.2,  # m
        "mild": 0.3,  # m
        "moderate": 0.4,  # m
    }

    stability_thresholds = {
        "normal": 8,  # Arbitrary score
        "mild": 6,
        "moderate": 4,
    }

    # Manual input grading
    manual_grades = {
        "Smoothly": 3,
        "Mild difficulty": 2,
        "Moderate difficulty": 1,
        "Severe difficulty or unable": 0,
    }

    # Determine grade based on metrics
    if (
        avg_gait_speed >= speed_thresholds["normal"]
        and deviation <= deviation_thresholds["normal"]
        and stability >= stability_thresholds["normal"]
    ):
        metrics_grade = 3
        metrics_explanation = (
            f"Normal gait speed ({avg_gait_speed:.2f} m/s), low deviation "
            f"({deviation:.2f} m), and high stability ({stability})."
        )
    elif (
        avg_gait_speed >= speed_thresholds["mild"]
        and deviation <= deviation_thresholds["mild"]
        and stability >= stability_thresholds["mild"]
    ):
        metrics_grade = 2
        metrics_explanation = (
            f"Mildly reduced gait speed ({avg_gait_speed:.2f} m/s), mild deviation "
            f"({deviation:.2f} m), and moderate stability ({stability})."
        )
    elif (
        avg_gait_speed >= speed_thresholds["moderate"]
        and deviation <= deviation_thresholds["moderate"]
        and stability >= stability_thresholds["moderate"]
    ):
        metrics_grade = 1
        metrics_explanation = (
            f"Moderate reduction in gait speed ({avg_gait_speed:.2f} m/s), moderate deviation "
            f"({deviation:.2f} m), and reduced stability ({stability})."
        )
    else:
        metrics_grade = 0
        metrics_explanation = (
            f"Severe issues: gait speed ({avg_gait_speed:.2f} m/s), deviation "
            f"({deviation:.2f} m), and stability ({stability})."
        )

    # Combine manual and metrics grades
    manual_grade = manual_grades.get(manual_input, 0)
    final_grade = min(metrics_grade, manual_grade)

    # Explanation
    explanation = (
        f"Metrics Assessment: {metrics_explanation}\n"
        f"Manual Input: '{manual_input}' indicates a grade of {manual_grade} for smoothness of head turns.\n"
        f"Final grade is {final_grade}."
    )

    return final_grade, explanation

def exercise04(metrics, manual_input):
    """
    Grades Exercise 04: Gait with Vertical Head Turns based on metrics and manual input.

    Parameters:
        metrics (dict): Dictionary containing automatic metrics (e.g., gait speed, stability).
        manual_input (str): Manual input describing the smoothness of vertical head turns.

    Returns:
        grade (int): Graded score (0-3) based on performance.
        explanation (str): Explanation of the grade.
    """
    # Default values for grade and explanation
    grade = 0
    explanation = "Insufficient data for grading."

    # Extract necessary metrics
    gait_speed = metrics.get("average_cadence", 0)  # Example: cadence as a proxy for gait speed
    stability_deviation = metrics.get("average_horizontal_distance", 0)  # Example: deviation
    time_to_complete = metrics.get("gait_cycles", [])[0].get("gait_cycle_duration", 0) if metrics.get("gait_cycles") else None

    # Grading logic
    if gait_speed > 60 and stability_deviation < 10 and time_to_complete < 5.5:
        automatic_grade = 3
        auto_explanation = (
            "Excellent performance: High cadence, minimal deviation, and fast completion."
        )
    elif 50 <= gait_speed <= 60 and 10 <= stability_deviation <= 20 and time_to_complete <= 7:
        automatic_grade = 2
        auto_explanation = (
            "Good performance: Moderate cadence, acceptable deviation, and reasonable completion time."
        )
    elif gait_speed < 50 or stability_deviation > 20 or time_to_complete > 7:
        automatic_grade = 1
        auto_explanation = (
            "Poor performance: Low cadence, high deviation, or slow completion."
        )
    else:
        automatic_grade = 0
        auto_explanation = (
            "Severe impairment: Metrics indicate significant issues with stability or speed."
        )

    # Manual input grading
    if manual_input == "Smoothly":
        manual_grade = 3
        manual_explanation = "Head turns performed smoothly."
    elif manual_input == "Mild difficulty":
        manual_grade = 2
        manual_explanation = "Mild difficulty in performing head turns."
    elif manual_input == "Moderate difficulty":
        manual_grade = 1
        manual_explanation = "Moderate difficulty in performing head turns."
    elif manual_input == "Severe difficulty or unable":
        manual_grade = 0
        manual_explanation = "Severe difficulty or unable to perform head turns."
    else:
        manual_grade = 0
        manual_explanation = "No valid manual input provided."

    # Combine grades
    grade = min(automatic_grade, manual_grade)
    explanation = (
        f"Automatic grading: {auto_explanation} Manual input grading: {manual_explanation}"
    )

    return grade, explanation

def exercise05(metrics, manual_input):
    """
    Grades the Gait and Pivot Turn exercise based on metrics and manual input.

    Parameters:
    - metrics (dict): Contains automatic metrics like "pivot_time", "balance_deviation", etc.
    - manual_input (str): Smoothness of pivot turn (e.g., "Smooth and balanced", "Mild imbalance", etc.).

    Returns:
    - grade (int): The grade for the exercise (0 to 3).
    - explanation (str): Explanation of the grade.
    """
    # Extract relevant metrics
    pivot_time = metrics.get("pivot_time", None)  # Time to complete the pivot in seconds
    balance_deviation = metrics.get("balance_deviation", None)  # Balance deviation during the pivot (normalized)
    
    # Define grading criteria
    if pivot_time is None or balance_deviation is None:
        return 0, "Insufficient data: Missing pivot time or balance deviation metrics."

    if pivot_time <= 2.0 and balance_deviation <= 0.1 and manual_input == "Smooth and balanced":
        grade = 3
        explanation = (
            f"Grade 3: The pivot was completed in {pivot_time:.2f} seconds with minimal balance deviation "
            f"({balance_deviation:.2f}), and the movement was smooth and balanced."
        )
    elif pivot_time <= 3.0 and balance_deviation <= 0.2 and manual_input in ["Smooth and balanced", "Mild imbalance"]:
        grade = 2
        explanation = (
            f"Grade 2: The pivot took {pivot_time:.2f} seconds with moderate balance deviation "
            f"({balance_deviation:.2f}), and the movement showed mild imbalance."
        )
    elif pivot_time <= 4.0 and balance_deviation <= 0.3 and manual_input in [
        "Mild imbalance",
        "Significant imbalance or hesitation",
    ]:
        grade = 1
        explanation = (
            f"Grade 1: The pivot took {pivot_time:.2f} seconds with significant balance deviation "
            f"({balance_deviation:.2f}), and the movement showed notable imbalance or hesitation."
        )
    else:
        grade = 0
        explanation = (
            f"Grade 0: The pivot took {pivot_time:.2f} seconds with excessive balance deviation "
            f"({balance_deviation:.2f}), or the movement was unable to complete successfully."
        )

    return grade, explanation


def exercise06(metrics, manual_input):
    """
    Grade Exercise 06: Step Over Obstacle.

    Args:
        metrics (dict): The automatic metrics for the exercise.
        manual_input (str): The manual input rating for smoothness.

    Returns:
        tuple: A grade and an explanation.
    """
    try:
        # Extract relevant metrics
        gait_speed_before = metrics.get("gait_speed_before", 0)
        gait_speed_during = metrics.get("gait_speed_during", 0)
        gait_speed_after = metrics.get("gait_speed_after", 0)
        deviation = metrics.get("deviation", 0)
        step_height = metrics.get("step_height", 0)

        # Analyze metrics
        if gait_speed_before > 1.2 and gait_speed_during > 1.0 and gait_speed_after > 1.2:
            automatic_rating = "Good speed and stability."
            grade_auto = 3
        elif deviation < 15 and step_height > 0.2:
            automatic_rating = "Moderate performance, slight deviation."
            grade_auto = 2
        else:
            automatic_rating = "Significant issues with stability or deviation."
            grade_auto = 1

        # Combine automatic and manual grading
        manual_rating = {
            "Smooth": 3,
            "Slight hesitation": 2,
            "Significant effort or imbalance": 1,
            "Unable to perform": 0,
        }.get(manual_input, 0)

        # Final grade (weighted average, manual input more significant)
        final_grade = round((0.4 * grade_auto) + (0.6 * manual_rating))
        explanation = (
            f"Automatic Metrics: {automatic_rating}. "
            f"Manual Input: {manual_input}. "
            f"Final Grade: {final_grade}."
        )
        return final_grade, explanation

    except Exception as e:
        return 0, f"Error processing metrics: {str(e)}"

def exercise07(metrics):
    """
    Grade Exercise 07: Gait with Narrow Base of Support.

    Args:
        metrics (dict): The automatic metrics for the exercise.

    Returns:
        tuple: A grade and an explanation.
    """
    try:
        # Extract key metrics
        avg_horizontal_distance = metrics.get("average_horizontal_distance", 0)
        avg_vertical_distance = metrics.get("average_vertical_distance", 0)
        avg_cadence = metrics.get("average_cadence", 0)
        avg_double_support_time = metrics.get("average_double_support_time", 0)
        gait_cycles = metrics.get("gait_cycles", [])

        # Determine deviation grade
        if avg_horizontal_distance <= 10 and avg_vertical_distance <= 10:
            deviation_rating = "Minimal deviation."
            grade_deviation = 3
        elif avg_horizontal_distance <= 20 or avg_vertical_distance <= 20:
            deviation_rating = "Moderate deviation."
            grade_deviation = 2
        else:
            deviation_rating = "Significant deviation."
            grade_deviation = 1

        # Determine cadence consistency
        if avg_cadence >= 60:
            cadence_rating = "Stable cadence."
            grade_cadence = 3
        elif 50 <= avg_cadence < 60:
            cadence_rating = "Moderate stability in cadence."
            grade_cadence = 2
        else:
            cadence_rating = "Unstable cadence."
            grade_cadence = 1

        # Determine balance via double support time
        if avg_double_support_time <= 1.5:
            balance_rating = "Excellent balance."
            grade_balance = 3
        elif avg_double_support_time <= 2.5:
            balance_rating = "Moderate balance."
            grade_balance = 2
        else:
            balance_rating = "Significant instability."
            grade_balance = 1

        # Compute final grade
        final_grade = round((grade_deviation + grade_cadence + grade_balance) / 3)

        # Explanation
        explanation = (
            f"Deviation: {deviation_rating} "
            f"({avg_horizontal_distance} cm horizontal, {avg_vertical_distance} cm vertical). "
            f"Cadence: {cadence_rating} ({avg_cadence:.2f} steps/min). "
            f"Balance: {balance_rating} (average double support time: {avg_double_support_time:.2f} s). "
            f"Final Grade: {final_grade}."
        )

        return final_grade, explanation

    except Exception as e:
        return 0, f"Error processing metrics: {str(e)}"

def exercise08(metrics):
    """
    Grades the 'Gait with Eyes Closed' exercise based on metrics.

    Args:
        metrics (dict): Dictionary containing the metrics for the exercise.

    Returns:
        tuple: Grade (int) and explanation (str).
    """
    try:
        # Extract metrics
        average_horizontal_distance = metrics.get("average_horizontal_distance", None)
        average_vertical_distance = metrics.get("average_vertical_distance", None)
        number_of_steps = metrics.get("number_of_steps", None)
        time_to_walk = metrics.get("gait_cycles", [{}])[0].get("gait_cycle_duration", None) * number_of_steps

        # Ensure all required metrics are available
        if None in [average_horizontal_distance, average_vertical_distance, number_of_steps, time_to_walk]:
            return 0, "Insufficient data to calculate grade."

        # Define grading criteria
        if time_to_walk <= 7 and average_horizontal_distance <= 15 and average_vertical_distance <= 5:
            grade = 3
            explanation = (
                "Excellent performance: fast walking speed, minimal deviation in walking pattern, "
                "and stable gait during the task."
            )
        elif time_to_walk <= 10 and average_horizontal_distance <= 25 and average_vertical_distance <= 10:
            grade = 2
            explanation = (
                "Moderate performance: acceptable walking speed, slight deviation in walking pattern, "
                "and moderate stability."
            )
        elif time_to_walk <= 15 or average_horizontal_distance <= 40 or average_vertical_distance <= 15:
            grade = 1
            explanation = (
                "Impaired performance: slow walking speed, significant deviation in walking pattern, "
                "or noticeable instability during the task."
            )
        else:
            grade = 0
            explanation = (
                "Severely impaired: very slow walking speed, extreme deviation in walking pattern, "
                "or severe instability."
            )

    except Exception as e:
        grade = 0
        explanation = f"Error processing metrics: {str(e)}"

    return grade, explanation

def exercise09(metrics):
    """
    Grades the 'Ambulating Backwards' exercise based on metrics.

    Args:
        metrics (dict): Dictionary containing the metrics for the exercise.

    Returns:
        tuple: Grade (int) and explanation (str).
    """
    try:
        # Extract metrics
        average_cadence = metrics.get("average_cadence", None)
        average_horizontal_distance = metrics.get("average_horizontal_distance", None)
        average_vertical_distance = metrics.get("average_vertical_distance", None)
        gait_cycles = metrics.get("gait_cycles", [])

        # Ensure all required metrics are available
        if None in [average_cadence, average_horizontal_distance, average_vertical_distance] or not gait_cycles:
            return 0, "Insufficient data to calculate grade."

        # Analyze gait cycle consistency
        cadence_variation = max(
            cycle.get("cadence", 0) for cycle in gait_cycles
        ) - min(cycle.get("cadence", 0) for cycle in gait_cycles)

        # Define grading criteria
        if (
            average_cadence >= 50
            and average_horizontal_distance <= 15
            and average_vertical_distance <= 5
            and cadence_variation <= 5
        ):
            grade = 3
            explanation = (
                "Excellent performance: smooth and consistent backward walking with minimal deviations "
                "and stable balance."
            )
        elif (
            average_cadence >= 40
            and average_horizontal_distance <= 25
            and average_vertical_distance <= 10
            and cadence_variation <= 10
        ):
            grade = 2
            explanation = (
                "Moderate performance: acceptable backward walking with slight inconsistencies "
                "or mild instability."
            )
        elif (
            average_cadence >= 30
            or average_horizontal_distance <= 40
            or average_vertical_distance <= 15
        ):
            grade = 1
            explanation = (
                "Impaired performance: significant inconsistencies, deviation, or instability during "
                "backward walking."
            )
        else:
            grade = 0
            explanation = (
                "Severely impaired: unable to perform backward walking effectively due to very slow pace, "
                "extreme deviation, or severe instability."
            )

    except Exception as e:
        grade = 0
        explanation = f"Error processing metrics: {str(e)}"

    return grade, explanation

def exercise10(manual_inputs):
    """
    Grades the 'Steps' exercise based entirely on manual inputs.

    Args:
        manual_inputs (dict): Dictionary containing manual input values.

    Returns:
        tuple: Grade (int) and explanation (str).
    """
    try:
        smoothness = manual_inputs.get("smoothness")
        effort = manual_inputs.get("effort")
        balance = manual_inputs.get("balance")
        fatigue = int(manual_inputs.get("fatigue"))

        if None in [smoothness, effort, balance, fatigue]:
            return 0, "Incomplete manual input data provided."

        # Map smoothness to grade
        smoothness_map = {
            "Smooth and balanced": 3,
            "Mild difficulty": 2,
            "Significant imbalance": 1,
            "Unable to perform": 0,
        }
        grade = smoothness_map.get(smoothness, 0)

        # Adjust grade based on effort, balance, and fatigue
        if effort == "High" or balance in ["Moderately unstable", "Severely unstable"] or fatigue >= 8:
            grade -= 1
        if grade < 0:
            grade = 0

        explanation = (
            f"Smoothness: {smoothness}, Effort: {effort}, Balance: {balance}, Fatigue: {fatigue}. "
            "Grade reflects overall stepping performance considering stability and effort."
        )
    except Exception as e:
        grade = 0
        explanation = f"Error processing inputs: {str(e)}"

    return grade, explanation
