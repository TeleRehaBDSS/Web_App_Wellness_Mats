import pandas as pd
import numpy as np
import json
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys

@dataclass
class BasicMatSignals:
    time_s: np.ndarray
    force: np.ndarray
    area: np.ndarray
    frames: np.ndarray
    dt: float
    cop_x: np.ndarray
    cop_y: np.ndarray

def _parse_sensors_column(df: pd.DataFrame) -> np.ndarray:
    sensors_list = []
    # Try optimized parsing
    try:
        for raw in df["sensors"]:
            sensors_list.append(json.loads(raw))
    except:
        print("JSON parse failed, falling back to eval...", flush=True)
        import ast
        sensors_list = []
        for raw in df["sensors"]:
            sensors_list.append(ast.literal_eval(raw))
            
    return np.array(sensors_list, dtype=float)

def load_basic_signals(csv_path: str) -> BasicMatSignals:
    print(f"Loading {csv_path} (first 15000 rows)...", flush=True)
    df = pd.read_csv(csv_path, nrows=15000)
    
    if "timepoint" in df.columns:
        try:
            df["timepoint"] = pd.to_datetime(df["timepoint"])
            df = df.sort_values("timepoint").reset_index(drop=True)
            time = df["timepoint"]
            time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
        except:
             time_s = np.arange(len(df)) * 0.02
    else:
        time_s = np.arange(len(df)) * 0.02

    sensors = _parse_sensors_column(df)
    sensors[sensors < 2.0] = 0.0 # Noise
    
    force = sensors.sum(axis=(1, 2))
    area = (sensors > 1e-3).sum(axis=(1, 2))
    
    if len(time_s) > 1:
        dt = float(np.median(np.diff(time_s)))
    else:
        dt = 0.02
        
    return BasicMatSignals(time_s, force, area, sensors, dt, np.zeros_like(force), np.zeros_like(force))

def analyze_single_leg(signals: BasicMatSignals):
    force = signals.force
    area = signals.area
    frames = signals.frames
    active_mask = force > 10.0
    
    print(f"Total Frames: {len(force)}", flush=True)
    print(f"Active Frames: {np.sum(active_mask)}", flush=True)
    
    if not np.any(active_mask):
        print("No active frames found.", flush=True)
        return

    # Blob Counting Analysis
    single_blob_mask = np.zeros(len(frames), dtype=bool)
    blob_counts = np.zeros(len(frames), dtype=int)
    active_indices = np.where(active_mask)[0]
    
    valid_areas_single_blob = []
    
    print("\n--- Analyzing Frames (Sample) ---", flush=True)
    sample_indices = np.linspace(active_indices[0], active_indices[-1], 20, dtype=int)
    
    for i in active_indices:
        frame = frames[i]
        if np.max(frame) > 0:
            # Debug sensor values
            if i in sample_indices:
                print(f"Frame {i}: Max Sensor Val={np.max(frame)}", flush=True)

            norm_frame = (frame / np.max(frame) * 255).astype(np.uint8)
            _, thresh = cv2.threshold(norm_frame, 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Debug blob sizes
            blob_sizes = [cv2.contourArea(cnt) for cnt in contours]
            significant_blobs = [s for s in blob_sizes if s > 10]
            n_blobs = len(significant_blobs)
            
            blob_counts[i] = n_blobs
            
            if n_blobs == 1:
                single_blob_mask[i] = True
                valid_areas_single_blob.append(area[i])
                
            if i in sample_indices:
                print(f"Frame {i}: Time={signals.time_s[i]:.2f}s, Area={area[i]}, Blobs={n_blobs}, Sizes={blob_sizes}", flush=True)

    # Refinement
    final_mask = single_blob_mask.copy()
    cutoff = 0
    median_area = 0
    
    if len(valid_areas_single_blob) > 0:
        median_area = np.median(valid_areas_single_blob)
        cutoff = 1.6 * median_area
        size_mask = area < cutoff
        final_mask = single_blob_mask & size_mask
        print(f"\nRefinement: Median Single Area={median_area:.1f}, Cutoff={cutoff:.1f}", flush=True)
        print(f"Frames removed by size filter: {np.sum(single_blob_mask & ~size_mask)}", flush=True)
    else:
        print("\nNo single-blob frames found!", flush=True)

    duration = np.sum(final_mask) * signals.dt
    print(f"\nCALCULATED DURATION: {duration:.2f} seconds", flush=True)
    
    # Plotting for Debug
    plt.figure(figsize=(12, 6))
    plt.plot(signals.time_s[active_mask], area[active_mask], 'k.', label='All Active', alpha=0.3)
    
    # Highlight phases
    plt.plot(signals.time_s[single_blob_mask], area[single_blob_mask], 'bo', label='1 Blob', alpha=0.5)
    plt.plot(signals.time_s[final_mask], area[final_mask], 'g.', label='Final Single Leg', markersize=5)
    
    if cutoff > 0:
        plt.axhline(cutoff, color='r', linestyle='--', label='Size Cutoff')
        plt.axhline(median_area, color='g', linestyle=':', label='Median Single Area')
        
    plt.legend()
    plt.title(f"Single Leg Detection Debug\nDuration: {duration:.2f}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Contact Area (pixels)")
    plt.savefig("debug_single_leg.png")
    print("Saved debug plot to debug_single_leg.png", flush=True)

if __name__ == "__main__":
    file_path = "MiniBEST_Web/data/uploads/1/1/Evita_Stand_on_one_leg_20250708_122026(RIGHT LEG OFF).csv"
    
    import os
    print(f"Current CWD: {os.getcwd()}", flush=True)
    
    if not os.path.exists(file_path):
        print(f"File not found at {file_path}, searching...", flush=True)
        for root, dirs, files in os.walk("."):
            for file in files:
                if "Evita_Stand_on_one_leg" in file and "RIGHT LEG OFF" in file:
                    file_path = os.path.join(root, file)
                    print(f"Found: {file_path}", flush=True)
                    break
    
    if os.path.exists(file_path):
        print(f"Analyzing: {file_path}", flush=True)
        s = load_basic_signals(file_path)
        analyze_single_leg(s)
    else:
        print("Could not find test file.", flush=True)