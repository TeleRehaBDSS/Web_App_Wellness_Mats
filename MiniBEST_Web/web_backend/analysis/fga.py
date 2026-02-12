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
