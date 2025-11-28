import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import cv2
from main_MINIBEST import (
    load_basic_signals,
    process_compensatory_stepping,
    process_rise_to_toes,
    process_sit_to_stand,
    process_stand_on_one_leg,
    process_stance_eyes_open,
    process_stance_eyes_closed,
    _infer_participant_and_exercise,
    BasicMatSignals,
    _get_blobs
)

class ModernMiniBESTGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MiniBEST Mat Analysis Dashboard")
        self.geometry("1400x950")
        
        # Theme configuration
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except:
            pass
            
        self._configure_styles()

        # Application State
        self.current_csv_path = None
        self.current_signals = None
        self.current_result = None

        # Replay state (for stance exercises and others that use frame-by-frame video)
        self.replay_frames = None
        self.replay_ax = None
        self.replay_im = None
        self.replay_running = False
        self.replay_after_id = None
        self.replay_frame_index = 0
        self.replay_start_index = 0
        self.replay_skip = 1  # how many frames to advance per step (set for stance)
        self.replay_indices = None  # optional list of frame indices to use (e.g., only active frames)
        self.replay_time = None     # corresponding time stamps for frames
        self.replay_start_time = 0.0
        self.animation = None
        
        # Main Layout
        self._create_layout()
        
    def _configure_styles(self):
        # Colors
        self.bg_color = "#f5f6f7"
        self.panel_bg = "#ffffff"
        self.accent_color = "#0078d4" # Modern blue
        self.text_primary = "#201f1e"
        self.text_secondary = "#605e5c"
        
        self.configure(bg=self.bg_color)
        
        # General Styles
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Card.TFrame", background=self.panel_bg, relief="flat")
        
        # Labels
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_primary, font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background=self.panel_bg, foreground=self.text_primary, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=self.bg_color, foreground=self.accent_color, font=("Segoe UI", 18, "bold"))
        self.style.configure("SubHeader.TLabel", background=self.panel_bg, foreground=self.text_secondary, font=("Segoe UI", 12, "bold"))
        
        # Score Display
        self.style.configure("ScoreTitle.TLabel", background=self.panel_bg, foreground=self.text_secondary, font=("Segoe UI", 14))
        self.style.configure("ScoreValue.TLabel", background=self.panel_bg, foreground=self.accent_color, font=("Segoe UI", 64, "bold"))
        
        # Buttons
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        self.style.map("Accent.TButton", background=[('active', "#006cc1"), ('!disabled', self.accent_color)], foreground=[('!disabled', 'white')])

        # Treeview
        self.style.configure("Treeview", rowheight=30, font=("Segoe UI", 10), background="white", fieldbackground="white")
        self.style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _create_layout(self):
        # Use a PanedWindow for resizable layout
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT SIDEBAR (Controls & Score) ---
        self.sidebar = ttk.Frame(self.paned, width=350)
        self.paned.add(self.sidebar, weight=1)
        
        # Input Card
        self.input_card = ttk.Frame(self.sidebar, style="Card.TFrame", padding=20)
        self.input_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.input_card, text="Analysis Setup", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 15))
        
        # File Selection
        ttk.Label(self.input_card, text="Data File (CSV):", style="Card.TLabel").pack(anchor="w")
        self.file_frame = ttk.Frame(self.input_card, style="Card.TFrame")
        self.file_frame.pack(fill=tk.X, pady=(5, 15))
        
        self.lbl_filename = ttk.Label(self.file_frame, text="No file selected", style="Card.TLabel", wraplength=280, foreground="gray")
        self.lbl_filename.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.file_frame, text="Browse...", command=self.browse_file).pack(side=tk.RIGHT)
        
        # Exercise Selection
        ttk.Label(self.input_card, text="Exercise Type:", style="Card.TLabel").pack(anchor="w")
        self.exercise_var = tk.StringVar()
        self.combo_exercise = ttk.Combobox(self.input_card, textvariable=self.exercise_var, state="readonly", height=15)
        self.combo_exercise['values'] = [
            "Compensatory Stepping (FORWARD)",
            "Compensatory Stepping (BACKWARD)",
            "Compensatory Stepping (LATERAL-LEFT)",
            "Compensatory Stepping (LATERAL-RIGHT)",
            "Rise to Toes",
            "Sit to Stand",
            "Stand on One Leg (Left)",
            "Stand on One Leg (Right)",
            "Stance (Eyes Open)",
            "Stance (Eyes Closed)",
        ]
        self.combo_exercise.pack(fill=tk.X, pady=(5, 15))
        self.combo_exercise.bind("<<ComboboxSelected>>", self._on_exercise_change)
        
        # --- Sit to Stand Details Frame (Initially Hidden) ---
        self.frame_sit_stand_opts = ttk.Frame(self.input_card, style="Card.TFrame")
        self.var_used_hands = tk.BooleanVar(value=False)
        self.chk_used_hands = ttk.Checkbutton(self.frame_sit_stand_opts, text="Used Hands?", variable=self.var_used_hands, style="Card.TCheckbutton")
        self.chk_used_hands.pack(anchor="w")
        
        self.var_multiple_attempts = tk.BooleanVar(value=False)
        self.chk_multiple_attempts = ttk.Checkbutton(self.frame_sit_stand_opts, text="Multiple Attempts / Assistance?", variable=self.var_multiple_attempts, style="Card.TCheckbutton")
        self.chk_multiple_attempts.pack(anchor="w")
        # -----------------------------------------------------

        # Participant
        ttk.Label(self.input_card, text="Participant ID:", style="Card.TLabel").pack(anchor="w")
        self.entry_participant = ttk.Entry(self.input_card)
        self.entry_participant.pack(fill=tk.X, pady=(5, 20))
        
        # Analyze Button
        self.btn_analyze = ttk.Button(self.input_card, text="ANALYZE EXERCISE", style="Accent.TButton", command=self.on_analyze, cursor="hand2")
        self.btn_analyze.pack(fill=tk.X, ipady=10)
        
        # Score Card
        self.score_card = ttk.Frame(self.sidebar, style="Card.TFrame", padding=20)
        self.score_card.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.score_card, text="MiniBEST Score", style="ScoreTitle.TLabel").pack(anchor="center")
        self.lbl_score = ttk.Label(self.score_card, text="-", style="ScoreValue.TLabel")
        self.lbl_score.pack(anchor="center", pady=10)
        self.lbl_score_desc = ttk.Label(self.score_card, text="", style="Card.TLabel", foreground="gray")
        self.lbl_score_desc.pack(anchor="center")

        # --- RIGHT MAIN AREA (Metrics & Plots) ---
        self.main_area = ttk.Frame(self.paned)
        self.paned.add(self.main_area, weight=4)
        
        # Notebook for Tabs
        self.notebook = ttk.Notebook(self.main_area)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Dashboard (Plots + Key Metrics)
        self.tab_dashboard = ttk.Frame(self.notebook, style="Card.TFrame", padding=10)
        self.notebook.add(self.tab_dashboard, text="  Visual Analysis  ")
        
        # Split Dashboard: Top for Metrics Table, Bottom for Plots
        self.dash_paned = ttk.PanedWindow(self.tab_dashboard, orient=tk.VERTICAL)
        self.dash_paned.pack(fill=tk.BOTH, expand=True)
        
        # Metrics Table Area
        self.metrics_frame = ttk.Frame(self.dash_paned, style="Card.TFrame")
        self.dash_paned.add(self.metrics_frame, weight=1)
        
        self.tree_metrics = ttk.Treeview(self.metrics_frame, columns=("Value",), show="tree headings")
        self.tree_metrics.heading("#0", text="Feature", anchor="w")
        self.tree_metrics.heading("Value", text="Value", anchor="w")
        self.tree_metrics.column("#0", width=300)
        self.tree_metrics.column("Value", width=150)
        self.tree_metrics.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plot Area
        self.plot_frame = ttk.Frame(self.dash_paned, style="Card.TFrame")
        self.dash_paned.add(self.plot_frame, weight=3)
        
        # This frame will hold the replay buttons, and is packed at the bottom.
        self.replay_controls = ttk.Frame(self.plot_frame, style="Card.TFrame")
        self.replay_controls.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        # This frame will hold the plot canvas, and is packed above the controls.
        canvas_container = ttk.Frame(self.plot_frame, style="Card.TFrame")
        canvas_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Initialize empty plot and add it to the canvas container
        self.fig = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create replay buttons inside their dedicated frame. They will be hidden/shown as needed.
        self.btn_replay_play = ttk.Button(
            self.replay_controls, text="Play Stance Video", command=self.start_replay
        )
        self.btn_replay_pause = ttk.Button(
            self.replay_controls, text="Pause Video", command=self.stop_replay
        )
        # The buttons are not packed here; they are packed in _update_plots.


    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            self.current_csv_path = path
            self.lbl_filename.config(text=os.path.basename(path), foreground="black")
            
            # Auto-detect
            folder_name = os.path.basename(os.path.dirname(path))
            self.entry_participant.delete(0, tk.END)
            self.entry_participant.insert(0, folder_name)
            
            _, ex_key, variant = _infer_participant_and_exercise(os.path.dirname(path), os.path.basename(path))
            if ex_key:
                self._select_exercise_from_key(ex_key, variant)

    def _select_exercise_from_key(self, key, variant):
        mapping = {
            "comp_stepping_forward": "Compensatory Stepping (FORWARD)",
            "comp_stepping_backward": "Compensatory Stepping (BACKWARD)",
            "comp_stepping_lateral_left": "Compensatory Stepping (LATERAL-LEFT)",
            "comp_stepping_lateral_right": "Compensatory Stepping (LATERAL-RIGHT)",
            "rise_to_toes": "Rise to Toes",
            "sit_to_stand": "Sit to Stand",
            "stand_one_leg_left": "Stand on One Leg (Left)",
            "stand_one_leg_right": "Stand on One Leg (Right)",
            "stance_eyes_open": "Stance (Eyes Open)",
            "stance_eyes_closed": "Stance (Eyes Closed)",
        }
        
        if key in mapping:
            self.combo_exercise.set(mapping[key])
            # Manually trigger the change event handler to update the UI (show/hide options)
            self._on_exercise_change(None)
    
    def _on_exercise_change(self, event):
        self.lbl_score.config(text="-")
        self.lbl_score_desc.config(text="")
        
        ex = self.combo_exercise.get()
        if "Sit to Stand" in ex:
            self.frame_sit_stand_opts.pack(fill=tk.X, pady=(5, 15), after=self.combo_exercise)
        else:
            self.frame_sit_stand_opts.pack_forget()

    def on_analyze(self):
        if not self.current_csv_path:
            messagebox.showwarning("Input Error", "Please select a CSV file first.")
            return
            
        ex_selection = self.combo_exercise.get()
        if not ex_selection:
            messagebox.showwarning("Input Error", "Please select an exercise type.")
            return
            
        participant = self.entry_participant.get().strip() or "Unknown"
        
        try:
            # Process based on selection
            if "Compensatory Stepping" in ex_selection:
                if "FORWARD" in ex_selection: direction = "FORWARD"
                elif "BACKWARD" in ex_selection: direction = "BACKWARD"
                elif "LATERAL-LEFT" in ex_selection: direction = "LATERAL-LEFT"
                elif "LATERAL-RIGHT" in ex_selection: direction = "LATERAL-RIGHT"
                else: direction = "UNKNOWN"
                
                self.current_result = process_compensatory_stepping(self.current_csv_path, direction, participant)
                
            elif "Rise to Toes" in ex_selection:
                self.current_result = process_rise_to_toes(self.current_csv_path, participant)
                
            elif "Sit to Stand" in ex_selection:
                # Pass clinician input
                used_hands = self.var_used_hands.get()
                multiple_attempts = self.var_multiple_attempts.get()
                self.current_result = process_sit_to_stand(self.current_csv_path, participant, used_hands, multiple_attempts)
                
            elif "Stand on One Leg" in ex_selection:
                leg = "Left" if "Left" in ex_selection else "Right"
                self.current_result = process_stand_on_one_leg(self.current_csv_path, participant, leg)
                
            elif "Stance (Eyes Open)" in ex_selection:
                self.current_result = process_stance_eyes_open(self.current_csv_path, participant)
                
            elif "Stance (Eyes Closed)" in ex_selection:
                self.current_result = process_stance_eyes_closed(self.current_csv_path, participant)
            
            # Load signals for plotting
            self.current_signals = load_basic_signals(self.current_csv_path)
            
            # Update UI
            self._update_score_display()
            self._update_metrics_table()
            self._update_plots(ex_selection)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_score_display(self):
        if self.current_result:
            score = self.current_result.score
            self.lbl_score.config(text=str(score))
            
            if score == 2: desc = "Normal"
            elif score == 1: desc = "Moderate"
            else: desc = "Severe / Unable"
            self.lbl_score_desc.config(text=desc)

    def _update_metrics_table(self):
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)
            
        if self.current_result:
            features = self.current_result.features
            for k, v in features.items():
                if k.startswith("_"): continue
                if isinstance(v, float):
                    val_str = f"{v:.3f}"
                else:
                    val_str = str(v)
                self.tree_metrics.insert("", "end", text=k, values=(val_str,))

    def _update_plots(self, ex_type):
        # Stop any running animation
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

        # Reset any running replay whenever plots are refreshed
        if self.replay_after_id is not None:
            try:
                self.after_cancel(self.replay_after_id)
            except Exception:
                pass
            self.replay_after_id = None
        self.replay_running = False
        self.replay_frames = None
        self.replay_indices = None
        self.replay_ax = None
        self.replay_im = None

        self.fig.clear()
        s = self.current_signals
        features = self.current_result.features
        
        # --- Hide/Show Replay Buttons Based on Exercise ---
        # First, always clear any existing buttons from the control frame
        for widget in self.replay_controls.winfo_children():
            widget.pack_forget()

        # Different layouts for different exercises
        if "Rise to Toes" in ex_type:
            # Show replay controls clearly under the plot area
            self.btn_replay_play.pack(side=tk.LEFT, padx=(0, 5))
            self.btn_replay_pause.pack(side=tk.LEFT)
            self._plot_rise_to_toes(s, features)
        elif "Compensatory Stepping" in ex_type:
            self._plot_compensatory(s, features)
        elif "Sit to Stand" in ex_type:
            # Enable replay for Sit to Stand
            self.btn_replay_play.pack(side=tk.LEFT, padx=(0, 5))
            self.btn_replay_pause.pack(side=tk.LEFT)
            self._plot_sit_to_stand(s, features)
        elif "Stance (Eyes Open)" in ex_type or "Stance (Eyes Closed)" in ex_type:
            # For stance, pack the buttons into the now-visible controls frame
            self.btn_replay_play.pack(side=tk.LEFT, padx=(0, 5))
            self.btn_replay_pause.pack(side=tk.LEFT)
            self._plot_stance(s, features)
        else:
            self._plot_general_sway(s, features)
            
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_rise_to_toes(self, s, features):
        """
        Visualizations for Rise to Toes:
        1. Video Replay (Top-Left)
        2. Contact Area vs Time (Top-Right) - shows the heel rise
        3. Anterior-Posterior Balance vs Time (Bottom-Left)
        4. Foot Stability Map (Bottom-Right) - Contours + CoP
        """
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        
        # 1. Replay axis (Top-Left)
        ax_replay = self.fig.add_subplot(gs[0, 0])
        
        if s.frames.size > 0:
            force = s.force
            area = s.area
            # Use frames where we have meaningful contact
            active_mask = (force > 5.0) | (area > 0)
            active_idx = np.where(active_mask)[0]
            if active_idx.size > 0:
                self.replay_indices = active_idx.astype(int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
            else:
                self.replay_indices = np.arange(len(s.frames), dtype=int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
        else:
            self.replay_start_index = 0
            first_frame = np.zeros((10, 10))
            self.replay_time = None
            self.replay_start_time = 0.0
            
        im_replay = ax_replay.imshow(first_frame, cmap='hot', interpolation='nearest', aspect='auto')
        ax_replay.set_title("Pressure Map (Replay)")
        ax_replay.axis("off")
        
        # Store for replay controls
        self.replay_frames = s.frames
        self.replay_ax = ax_replay
        self.replay_im = im_replay

        # 2. Contact Area (Top-Right) - Critical for identifying the heel rise
        ax_area = self.fig.add_subplot(gs[0, 1])
        
        valid_time = features.get("_valid_time")
        smoothed_area = features.get("_smoothed_area")
        
        if valid_time is not None and smoothed_area is not None and len(valid_time) == len(smoothed_area):
            ax_area.plot(valid_time, smoothed_area, 'b-', label='Contact Area')
        else:
            ax_area.plot(s.time_s, s.area, 'gray', alpha=0.5, label='Raw Area')
            
        baseline = features.get("Baseline Area (pixels)", 0)
        thresh = features.get("Area Threshold", 0)
        
        if baseline > 0:
            ax_area.axhline(y=baseline, color='k', linestyle='--', label='Flat Foot Baseline')
            ax_area.axhline(y=thresh, color='r', linestyle=':', label='Heel Rise Threshold')
            
        # Highlight the contiguous "on toes" regions
        on_toes_mask = features.get("_on_toes_mask")
        if on_toes_mask is not None and valid_time is not None:
             # We can shade the regions where on_toes_mask is True
             # Simple way: iterate runs
             is_up = False
             start_t = 0
             for i, up in enumerate(on_toes_mask):
                 if up and not is_up:
                     is_up = True
                     start_t = valid_time[i]
                 elif not up and is_up:
                     is_up = False
                     end_t = valid_time[i-1]
                     ax_area.axvspan(start_t, end_t, color='green', alpha=0.1)
             if is_up:
                 ax_area.axvspan(start_t, valid_time[-1], color='green', alpha=0.1)

        ax_area.set_title("Contact Area (Heel Rise Identification)")
        ax_area.set_ylabel("Area (pixels)")
        ax_area.set_xlabel("Time (s)")
        ax_area.legend(loc="upper right", fontsize='small')
        ax_area.grid(True, alpha=0.3)

        # 3. Forward-Backward Balance (CoP Y) vs Time (Bottom Left)
        ax_stab = self.fig.add_subplot(gs[1, 0])
        
        # Extract CoP Y (Anterior-Posterior) for active frames
        # active_mask aligns with s.cop_y
        force = s.force
        active_mask = force > 10.0 # Match criteria from main processing
        
        if np.any(active_mask):
            active_time = s.time_s[active_mask]
            active_cop_y = s.cop_y[active_mask]
            
            # Invert Y if necessary so "Up" on plot means "Forward/Toes"
            # Usually lower row index is "top" of mat, but let's assume standard Cartesian for plot.
            # Let's center it around 0 for clarity.
            cop_y_centered = active_cop_y - np.mean(active_cop_y[:10]) # Baseline from start
            
            ax_stab.plot(active_time, cop_y_centered, 'purple', label='Forward-Backward Shift')
            
            # Visualize the "On Toes" phase if available
            on_toes_mask = features.get("_on_toes_mask")
            valid_time = features.get("_valid_time") # This should align with active_mask roughly
            
            if on_toes_mask is not None and valid_time is not None and len(valid_time) == len(active_time):
                 # Shade the regions where on_toes_mask is True
                 is_up = False
                 start_t = 0
                 for i, up in enumerate(on_toes_mask):
                     if up and not is_up:
                         is_up = True
                         start_t = valid_time[i]
                     elif not up and is_up:
                         is_up = False
                         end_t = valid_time[i-1]
                         ax_stab.axvspan(start_t, end_t, color='green', alpha=0.1, label='On Toes Phase' if start_t==valid_time[0] else "")
                 if is_up:
                     ax_stab.axvspan(start_t, valid_time[-1], color='green', alpha=0.1)

        ax_stab.set_title("Anterior-Posterior Balance (Forward Shift to Toes)")
        ax_stab.set_ylabel("CoP Y Displacement (sensor units)")
        ax_stab.set_xlabel("Time (s)")
        # ax_stab.legend(loc="upper right") # Legend can be clutter if multiple spans
        ax_stab.grid(True, alpha=0.3)

        # 4. Foot Stability Map (Bottom Right)
        # Standard View (No Rotation): X=Medial-Lateral, Y=Anterior-Posterior
        ax_stab_map = self.fig.add_subplot(gs[1, 1])
        
        if np.any(active_mask):
            # 1. Calculate everything in ORIGINAL coordinates first
            frames_active = s.frames[active_mask]
            force_active = s.force[active_mask]
            max_force = np.max(force_active) if len(force_active) > 0 else 1.0
            
            # Average frame for contours
            avg_frame = np.mean(frames_active, axis=0) # Shape (H, W)
            original_h, original_w = avg_frame.shape
            
            # Normalize for contour detection
            if np.max(avg_frame) > 0:
                norm_avg = (avg_frame / np.max(avg_frame) * 255).astype(np.uint8)
            else:
                norm_avg = avg_frame.astype(np.uint8)
                
            _, thresh_avg = cv2.threshold(norm_avg, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            valid_blobs = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        valid_blobs.append({'contour': cnt, 'centroid': (cx, cy), 'area': area})
            
            # Identify Left vs Right in Original Coords
            mask_L = np.zeros_like(avg_frame, dtype=bool)
            mask_R = np.zeros_like(avg_frame, dtype=bool)
            contour_L = None
            contour_R = None
            
            if len(valid_blobs) >= 2:
                foot1 = valid_blobs[0]
                foot2 = valid_blobs[1]
                if foot1['centroid'][0] < foot2['centroid'][0]:
                    blob_L, blob_R = foot1, foot2
                else:
                    blob_L, blob_R = foot2, foot1
                    
                contour_L = blob_L['contour']
                contour_R = blob_R['contour']
                
                m_L = np.zeros((original_h, original_w), dtype=np.uint8)
                m_R = np.zeros((original_h, original_w), dtype=np.uint8)
                cv2.drawContours(m_L, [contour_L], -1, 1, -1)
                cv2.drawContours(m_R, [contour_R], -1, 1, -1)
                mask_L = m_L.astype(bool)
                mask_R = m_R.astype(bool)
            else:
                midline = original_w // 2
                mask_L[:, :midline] = True
                mask_R[:, midline:] = True
            
            # Calculate Centroids in Original Coords
            def get_centroids_masked(frames_stack, mask):
                masked_frames = frames_stack * mask[None, :, :]
                weights = np.sum(masked_frames, axis=(1, 2))
                valid = weights > 1.0 
                weights[~valid] = 1.0 
                rows, c = frames_stack.shape[1], frames_stack.shape[2]
                r_idx, c_idx = np.indices((rows, c))
                cy = np.sum(masked_frames * r_idx[None, :, :], axis=(1, 2)) / weights
                cx = np.sum(masked_frames * c_idx[None, :, :], axis=(1, 2)) / weights
                return cx, cy, valid

            lx, ly, l_valid = get_centroids_masked(frames_active, mask_L)
            rx, ry, r_valid = get_centroids_masked(frames_active, mask_R)

            # 1. Data-Driven Smoothed Contour (Tight Fit)
            # Instead of a generic template, we create a smooth hull around the actual pressure data.
            
            def get_smooth_contour(contour):
                """
                Generates a smooth, tight polygon around the raw contour.
                """
                if contour is None or len(contour) < 3:
                    return None
                
                # 1. Get Convex Hull to wrap the points tightly
                hull = cv2.convexHull(contour)
                
                # 2. Smooth the hull using simple corner averaging (Chaikin's Algorithm style)
                if len(hull) < 3: 
                    return hull.reshape(-1, 2)
                
                # Extract points
                pts = hull.reshape(-1, 2)
                
                # Simple smoothing: averaging neighbors
                # Repeat first few points for wrap-around
                pts_wrap = np.vstack([pts, pts[:2]])
                
                smooth_pts = []
                for i in range(len(pts)):
                    # Weighted average of Prev, Curr, Next
                    p_prev = pts_wrap[i]
                    p_curr = pts_wrap[i+1]
                    p_next = pts_wrap[i+2]
                    
                    new_p = 0.25*p_prev + 0.5*p_curr + 0.25*p_next
                    smooth_pts.append(new_p)
                    
                return np.array(smooth_pts)

            def draw_tight_contour(ax, contour, is_left):
                smooth_poly = get_smooth_contour(contour)
                if smooth_poly is None:
                    return
                
                color = '#B0C4DE' if is_left else '#F08080' # LightSteelBlue / LightCoral
                edge_color = '#4070a0' if is_left else '#a04040'
                
                ax.fill(smooth_poly[:, 0], smooth_poly[:, 1], color=color, alpha=0.4, label=f"{'Left' if is_left else 'Right'} Foot Area")
                ax.plot(smooth_poly[:, 0], smooth_poly[:, 1], color=edge_color, linewidth=2, alpha=0.8)

            # Draw Left Foot
            if contour_L is not None:
                draw_tight_contour(ax_stab_map, contour_L, True)
                
            # Draw Right Foot
            if contour_R is not None:
                draw_tight_contour(ax_stab_map, contour_R, False)
            
            # 3. Plot Scatter Points (Direct Centroids, No Rotation)
            
            # Determine "On Toes" vs "Flat" based on Area
            # We can use the features if available, or calculate locally for the active frames
            area_active = s.area[active_mask]
            
            # Use the threshold computed in analysis, or fallback
            thresh_area = features.get("Area Threshold", np.max(area_active) * 0.6)
            
            # Identify phases
            is_on_toes = area_active < thresh_area
            is_flat = ~is_on_toes
            
            # Legend Helper - Explicitly define all colors for presentation quality
            # Left Foot
            ax_stab_map.scatter([], [], c='blue', s=30, label='L: Flat (Blue)')
            ax_stab_map.scatter([], [], c='red', s=30, label='L: Toes (Red)')
            # Right Foot
            ax_stab_map.scatter([], [], c='cyan', s=30, label='R: Flat (Cyan)')
            ax_stab_map.scatter([], [], c='magenta', s=30, label='R: Toes (Magenta)')
            # Avg Marker
            ax_stab_map.plot([], [], 'k+', markersize=10, markeredgewidth=2, label='Avg Peak (+)')
            
            # Left Foot Plotting
            l_valid_mask = l_valid # Points where left foot is detected
            
            # Flat Phase (Blue)
            l_flat = l_valid_mask & is_flat
            if np.any(l_flat):
                ax_stab_map.scatter(lx[l_flat], ly[l_flat], c='blue', s=10, alpha=0.3)
                
            # Toes Phase (Red)
            l_toes = l_valid_mask & is_on_toes
            if np.any(l_toes):
                ax_stab_map.scatter(lx[l_toes], ly[l_toes], c='red', s=15, alpha=0.8)
                # Mark average CoP on toes
                ax_stab_map.plot(np.mean(lx[l_toes]), np.mean(ly[l_toes]), 'k+', markersize=12, markeredgewidth=2)
            
            # Right Foot Plotting
            r_valid_mask = r_valid
            
            # Flat Phase (Cyan)
            r_flat = r_valid_mask & is_flat
            if np.any(r_flat):
                ax_stab_map.scatter(rx[r_flat], ry[r_flat], c='cyan', s=10, alpha=0.3)
                
            # Toes Phase (Magenta)
            r_toes = r_valid_mask & is_on_toes
            if np.any(r_toes):
                ax_stab_map.scatter(rx[r_toes], ry[r_toes], c='magenta', s=15, alpha=0.8)
                # Mark average CoP on toes
                ax_stab_map.plot(np.mean(rx[r_toes]), np.mean(ry[r_toes]), 'k+', markersize=12, markeredgewidth=2)

        ax_stab_map.set_title("Foot Stability Map (Rise to Toes)")
        ax_stab_map.set_xlabel("Medial-Lateral (X)")
        ax_stab_map.set_ylabel("Anterior-Posterior (Y)")
        # Use a smaller font or 2 columns to fit the detailed legend nicely
        ax_stab_map.legend(loc="best", fontsize='x-small', framealpha=0.8, ncol=2)
        ax_stab_map.axis('equal')
        ax_stab_map.set_axis_off()
        ax_stab_map.set_xlabel("Medial-Lateral (X)")
        ax_stab_map.set_ylabel("Anterior-Posterior (Y)")
        ax_stab_map.legend(loc="best", fontsize='x-small', framealpha=0.8)
        ax_stab_map.axis('equal')
        ax_stab_map.set_axis_off()

    def _plot_compensatory(self, s, features):
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1])
        
        # 1. Heatmap (Max Pressure to see steps)
        ax0 = self.fig.add_subplot(gs[0])
        max_frame = np.max(s.frames, axis=0) # Max over time shows "footprints"
        im = ax0.imshow(max_frame, cmap='viridis', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im, ax=ax0, label='Max Pressure')
        ax0.set_title("Max Pressure Map (Footprints Trace)")
        
        # 2. CoP Speed
        ax1 = self.fig.add_subplot(gs[1])
        dt = max(s.dt, 1e-4)
        speed = np.sqrt(np.diff(s.cop_x)**2 + np.diff(s.cop_y)**2) / dt
        ax1.plot(s.time_s[:-1], speed, 'b-')
        ax1.set_title("CoP Speed (Step Impulse)")
        ax1.set_ylabel("Speed")
        ax1.grid(True, alpha=0.3)
        
        # 3. CoP Trajectory
        ax2 = self.fig.add_subplot(gs[2])
        ax2.plot(s.cop_x, s.cop_y, 'k-', alpha=0.6)
        ax2.plot(s.cop_x[0], s.cop_y[0], 'go', label='Start')
        ax2.plot(s.cop_x[-1], s.cop_y[-1], 'rx', label='End')
        ax2.set_title("CoP Trajectory")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.axis('equal')
        ax2.legend()

    def _plot_sit_to_stand(self, s, features):
        """
        Visualizations for Sit to Stand:
        1. Video Replay (Top-Left)
        2. Total Force Profile (Top-Right) - shows loading/unloading and rise
        3. CoP Anterior-Posterior Displacement (Bottom Left) - Shows forward momentum
        4. CoP Trajectory / Butterfly (Bottom Right) - Shows the path and stability
        """
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1])
        
        # 1. Replay axis (Top-Left)
        ax_replay = self.fig.add_subplot(gs[0, 0])
        
        if s.frames.size > 0:
            active_idx = np.where(s.force > 10.0)[0]
            if active_idx.size > 0:
                self.replay_indices = active_idx.astype(int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
            else:
                self.replay_indices = np.arange(len(s.frames), dtype=int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
        else:
            self.replay_start_index = 0
            first_frame = np.zeros((10, 10))
            self.replay_time = None
            self.replay_start_time = 0.0
            
        im_replay = ax_replay.imshow(first_frame, cmap='hot', interpolation='nearest', aspect='auto')
        ax_replay.set_title("Pressure Map (Replay)")
        ax_replay.axis("off")
        
        self.replay_frames = s.frames
        self.replay_ax = ax_replay
        self.replay_im = im_replay

        # 2. Total Force Profile (Top-Right)
        ax_force = self.fig.add_subplot(gs[0, 1])
        ax_force.plot(s.time_s, s.force, 'k-', label='Total Force')
        
        rise_start = features.get("Rise Start Time (s)")
        rise_end = features.get("Rise End Time (s)")
        
        if rise_start is not None and rise_end is not None and not np.isnan(rise_start):
            ax_force.axvline(x=rise_start, color='g', linestyle='--', label='Start')
            ax_force.axvline(x=rise_end, color='r', linestyle='--', label='Stable')
            ax_force.axvspan(rise_start, rise_end, color='green', alpha=0.1)
            
        ax_force.set_title("Total Force (Rise Detection)")
        ax_force.set_ylabel("Force (sum)")
        ax_force.set_xlabel("Time (s)")
        ax_force.legend(loc="best", fontsize='small')
        ax_force.grid(True, alpha=0.3)
        
        # 3. Anterior-Posterior Displacement (Bottom Left)
        ax_ap = self.fig.add_subplot(gs[1, 0])
        
        # Use active frames for CoP
        force = s.force
        active_mask = force > 10.0
        
        if np.any(active_mask):
            active_time = s.time_s[active_mask]
            active_cop_y = s.cop_y[active_mask]
            # Center around initial position
            cop_y_centered = active_cop_y - np.mean(active_cop_y[:10])
            
            ax_ap.plot(active_time, cop_y_centered, 'b-', label='AP Displacement')
            
            # Mark phases
            if rise_start is not None and rise_end is not None and not np.isnan(rise_start):
                 ax_ap.axvspan(rise_start, rise_end, color='green', alpha=0.1, label='Rise Phase')
                 
        ax_ap.set_title("Anterior-Posterior Shift (Momentum)")
        ax_ap.set_ylabel("CoP Y (Forward ->)")
        ax_ap.set_xlabel("Time (s)")
        ax_ap.legend(loc="best", fontsize='small')
        ax_ap.grid(True, alpha=0.3)

        # 4. Foot Stability Map (Bottom Right)
        # Shows exactly where the feet were and if they moved.
        # Standard View (No Rotation): X=Medial-Lateral, Y=Anterior-Posterior
        ax_stab_map = self.fig.add_subplot(gs[1, 1])
        
        if np.any(active_mask):
            # 1. Calculate everything in ORIGINAL coordinates first
            frames_active = s.frames[active_mask]
            force_active = s.force[active_mask]
            max_force = np.max(force_active) if len(force_active) > 0 else 1.0
            
            # Average frame for contours
            avg_frame = np.mean(frames_active, axis=0) # Shape (H, W)
            original_h, original_w = avg_frame.shape
            
            # Normalize for contour detection
            if np.max(avg_frame) > 0:
                norm_avg = (avg_frame / np.max(avg_frame) * 255).astype(np.uint8)
            else:
                norm_avg = avg_frame.astype(np.uint8)
                
            _, thresh_avg = cv2.threshold(norm_avg, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_avg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            valid_blobs = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        valid_blobs.append({'contour': cnt, 'centroid': (cx, cy), 'area': area})
            
            # Identify Left vs Right in Original Coords
            mask_L = np.zeros_like(avg_frame, dtype=bool)
            mask_R = np.zeros_like(avg_frame, dtype=bool)
            contour_L = None
            contour_R = None
            
            if len(valid_blobs) >= 2:
                foot1 = valid_blobs[0]
                foot2 = valid_blobs[1]
                if foot1['centroid'][0] < foot2['centroid'][0]:
                    blob_L, blob_R = foot1, foot2
                else:
                    blob_L, blob_R = foot2, foot1
                    
                contour_L = blob_L['contour']
                contour_R = blob_R['contour']
                
                m_L = np.zeros((original_h, original_w), dtype=np.uint8)
                m_R = np.zeros((original_h, original_w), dtype=np.uint8)
                cv2.drawContours(m_L, [contour_L], -1, 1, -1)
                cv2.drawContours(m_R, [contour_R], -1, 1, -1)
                mask_L = m_L.astype(bool)
                mask_R = m_R.astype(bool)
            else:
                midline = original_w // 2
                mask_L[:, :midline] = True
                mask_R[:, midline:] = True
            
            # Calculate Centroids in Original Coords
            def get_centroids_masked(frames_stack, mask):
                masked_frames = frames_stack * mask[None, :, :]
                weights = np.sum(masked_frames, axis=(1, 2))
                valid = weights > 1.0 
                weights[~valid] = 1.0 
                rows, c = frames_stack.shape[1], frames_stack.shape[2]
                r_idx, c_idx = np.indices((rows, c))
                cy = np.sum(masked_frames * r_idx[None, :, :], axis=(1, 2)) / weights
                cx = np.sum(masked_frames * c_idx[None, :, :], axis=(1, 2)) / weights
                return cx, cy, valid

            lx, ly, l_valid = get_centroids_masked(frames_active, mask_L)
            rx, ry, r_valid = get_centroids_masked(frames_active, mask_R)

            # 1. Data-Driven Smoothed Contour (Tight Fit)
            # Instead of a generic template, we create a smooth hull around the actual pressure data.
            
            def get_smooth_contour(contour):
                """
                Generates a smooth, tight polygon around the raw contour.
                """
                if contour is None or len(contour) < 3:
                    return None
                
                # 1. Get Convex Hull to wrap the points tightly
                hull = cv2.convexHull(contour)
                
                # 2. Smooth the hull using simple corner averaging (Chaikin's Algorithm style)
                # Or just use the hull directly if we want it very tight. 
                # A convex hull is already quite clean for a footprint.
                # Let's make it a bit organic by interpolating.
                
                if len(hull) < 3: 
                    return hull.reshape(-1, 2)
                
                # Extract points
                pts = hull.reshape(-1, 2)
                
                # Simple smoothing: averaging neighbors
                # Repeat first few points for wrap-around
                pts_wrap = np.vstack([pts, pts[:2]])
                
                smooth_pts = []
                for i in range(len(pts)):
                    # Weighted average of Prev, Curr, Next
                    # 0.25*Prev + 0.5*Curr + 0.25*Next
                    p_prev = pts_wrap[i]
                    p_curr = pts_wrap[i+1]
                    p_next = pts_wrap[i+2]
                    
                    new_p = 0.25*p_prev + 0.5*p_curr + 0.25*p_next
                    smooth_pts.append(new_p)
                    
                return np.array(smooth_pts)

            def draw_tight_contour(ax, contour, is_left):
                smooth_poly = get_smooth_contour(contour)
                if smooth_poly is None:
                    return
                
                color = '#B0C4DE' if is_left else '#F08080' # LightSteelBlue / LightCoral
                edge_color = '#4070a0' if is_left else '#a04040'
                
                ax.fill(smooth_poly[:, 0], smooth_poly[:, 1], color=color, alpha=0.4, label=f"{'Left' if is_left else 'Right'} Foot Area")
                ax.plot(smooth_poly[:, 0], smooth_poly[:, 1], color=edge_color, linewidth=2, alpha=0.8)

            # Draw Left Foot
            if contour_L is not None:
                draw_tight_contour(ax_stab_map, contour_L, True)
                
            # Draw Right Foot
            if contour_R is not None:
                draw_tight_contour(ax_stab_map, contour_R, False)
            
            # 3. Plot Scatter Points (Direct Centroids, No Rotation)
            is_standing = force_active > (0.8 * max_force)
            
            # Legend Helper
            ax_stab_map.scatter([], [], c='blue', label='Sit/Load')
            ax_stab_map.scatter([], [], c='red', label='Stand')
            
            # Left Foot Plotting
            l_sit_mask = l_valid & (~is_standing)
            if np.any(l_sit_mask):
                ax_stab_map.scatter(lx[l_sit_mask], ly[l_sit_mask], c='blue', s=3, alpha=0.3)
            
            l_stand_mask = l_valid & is_standing
            if np.any(l_stand_mask):
                ax_stab_map.scatter(lx[l_stand_mask], ly[l_stand_mask], c='red', s=8, alpha=0.8)
                # Average
                avg_lx = np.mean(lx[l_stand_mask])
                avg_ly = np.mean(ly[l_stand_mask])
                ax_stab_map.plot(avg_lx, avg_ly, 'k+', markersize=10, markeredgewidth=2)
                
            # Right Foot Plotting
            r_sit_mask = r_valid & (~is_standing)
            if np.any(r_sit_mask):
                ax_stab_map.scatter(rx[r_sit_mask], ry[r_sit_mask], c='cyan', s=3, alpha=0.3)
                
            r_stand_mask = r_valid & is_standing
            if np.any(r_stand_mask):
                ax_stab_map.scatter(rx[r_stand_mask], ry[r_stand_mask], c='magenta', s=8, alpha=0.8)
                avg_rx = np.mean(rx[r_stand_mask])
                avg_ry = np.mean(ry[r_stand_mask])
                ax_stab_map.plot(avg_rx, avg_ry, 'k+', markersize=10, markeredgewidth=2)

        ax_stab_map.set_title("Foot Stability Map (Standard View)\n(Sliding = Loading, Cluster = Stable)")
        ax_stab_map.set_xlabel("Medial-Lateral (X)")
        ax_stab_map.set_ylabel("Anterior-Posterior (Y)")
        ax_stab_map.legend(loc="best", fontsize='x-small', framealpha=0.8)
        ax_stab_map.axis('equal')
        ax_stab_map.axis('off')


    def _plot_stance(self, s: BasicMatSignals, features):
        """
        Stance-specific visualization (eyes open / eyes closed):
        - Top-left: live replay of frames (used by the replay controls)
        - Top-right: average pressure map over the stance
        - Bottom: CoP trajectory on the mat
        """
        # 2x2 grid: replay | average heatmap
        #           CoP trajectory (full width)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

        # 1. Replay axis (this will be updated by the replay controls)
        ax_replay = self.fig.add_subplot(gs[0, 0])
        # Choose a sensible starting frame: first frame with meaningful force,
        # so that the video window is not just an empty black frame.
        if s.frames.size > 0:
            force = s.force
            area = s.area
            # Only use frames where we have meaningful contact, so feet are visible.
            # Criterion: total force above a very small threshold OR area > 0.
            active_mask = (force > 5.0) | (area > 0)
            active_idx = np.where(active_mask)[0]
            if active_idx.size > 0:
                # Use only active frames for the stance video
                self.replay_indices = active_idx.astype(int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
            else:
                # Fallback: use all frames if we cannot detect active ones
                self.replay_indices = np.arange(len(s.frames), dtype=int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
        else:
            self.replay_start_index = 0
            first_frame = np.zeros((10, 10))
            self.replay_time = None
            self.replay_start_time = 0.0
        im_replay = ax_replay.imshow(first_frame, cmap='hot', interpolation='nearest', aspect='auto')
        ax_replay.set_title("Pressure Map (Replay)")
        ax_replay.axis("off")

        # Store for replay controls
        self.replay_frames = s.frames
        self.replay_ax = ax_replay
        self.replay_im = im_replay

        # 2. Average pressure map
        ax_avg = self.fig.add_subplot(gs[0, 1])
        if s.frames.size > 0:
            avg_frame = np.mean(s.frames, axis=0)
        else:
            avg_frame = np.zeros((10, 10))
        im_avg = ax_avg.imshow(avg_frame, cmap='hot', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im_avg, ax=ax_avg, fraction=0.046, pad=0.04)
        ax_avg.set_title("Average Pressure Distribution (Stance)")
        ax_avg.axis("off")

        # 3. CoP Sway over Time (Medial-Lateral and Anterior-Posterior)
        ax_sway_time = self.fig.add_subplot(gs[1, 0])
        
        # Use only valid CoP data
        valid_mask = np.isfinite(s.cop_x) & np.isfinite(s.cop_y)
        if np.any(valid_mask):
            time_valid = s.time_s[valid_mask]
            cop_x_valid = s.cop_x[valid_mask]
            cop_y_valid = s.cop_y[valid_mask]
            
            # De-mean to show sway around the center
            cop_x_demeaned = cop_x_valid - np.mean(cop_x_valid)
            cop_y_demeaned = cop_y_valid - np.mean(cop_y_valid)

            ax_sway_time.plot(time_valid, cop_x_demeaned, 'r-', alpha=0.8, label='M/L Sway (X)')
            ax_sway_time.plot(time_valid, cop_y_demeaned, 'b-', alpha=0.8, label='A/P Sway (Y)')
        
        ax_sway_time.set_title("CoP Sway Over Time")
        ax_sway_time.set_xlabel("Time (s)")
        ax_sway_time.set_ylabel("Sway (sensor units)")
        ax_sway_time.legend(loc="best")
        ax_sway_time.grid(True, alpha=0.3)

        # 4. Sway Velocity over Time (Stability)
        ax_sway_vel = self.fig.add_subplot(gs[1, 1])
        
        dt = max(s.dt, 1e-4)
        vx = np.diff(s.cop_x) / dt
        vy = np.diff(s.cop_y) / dt
        speed = np.sqrt(vx**2 + vy**2)
        
        # Use time from the midpoint of diff
        time_for_speed = s.time_s[:-1]
        
        # Filter out extreme speeds from noise/gaps
        valid_speed_mask = (speed < 500) & np.isfinite(speed)
        
        if np.any(valid_speed_mask):
            ax_sway_vel.plot(time_for_speed[valid_speed_mask], speed[valid_speed_mask], 'purple', alpha=0.8, label='Sway Speed')

        ax_sway_vel.set_title("Postural Stability (Sway Speed)")
        ax_sway_vel.set_xlabel("Time (s)")
        ax_sway_vel.set_ylabel("Sway Speed (units/s)")
        ax_sway_vel.legend(loc="best")
        ax_sway_vel.grid(True, alpha=0.3)

    def _plot_general_sway(self, s, features):
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
        
        # Heatmap
        ax0 = self.fig.add_subplot(gs[0])
        avg_frame = np.mean(s.frames, axis=0)
        im = ax0.imshow(avg_frame, cmap='hot', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im, ax=ax0)
        ax0.set_title("Average Pressure Distribution")
        
        # Sway
        ax1 = self.fig.add_subplot(gs[1])
        ax1.plot(s.cop_x, s.cop_y, 'k-', alpha=0.6)
        ax1.set_title("Center of Pressure Trajectory")
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

    def _update_animation_frame(self, i):
        """Update function for FuncAnimation."""
        # i is the frame number from 0 to len(self.replay_indices)-1
        if not hasattr(self, 'replay_indices') or self.replay_indices is None or i >= len(self.replay_indices):
            return

        frame_idx = self.replay_indices[i]
        frame = self.replay_frames[frame_idx]
        self.replay_im.set_data(frame)

        # Update title with time
        if self.replay_time is not None:
            t_abs = float(self.replay_time[frame_idx])
            t_rel = max(0.0, t_abs - float(self.replay_start_time))
            self.replay_ax.set_title(f"Pressure Map (Replay)  t = {t_rel:.2f} s")

        # When blit=False, we don't return artists. The function just modifies the plot.
        return

    def _on_animation_end(self):
        """Callback for when the animation finishes."""
        # Set animation to None so we know it has finished.
        self.animation = None

    def start_replay(self):
        """Start replaying frames over time using FuncAnimation."""
        if self.replay_frames is None or self.replay_ax is None or not hasattr(self, 'replay_indices') or self.replay_indices is None:
            return

        # Stop any previous animation that might be running
        if self.animation is not None:
            try:
                if self.animation.event_source:
                    self.animation.event_source.stop()
            except AttributeError:
                pass
            self.animation = None

        # Calculate the average interval for a smooth playback that matches total duration
        if len(self.replay_indices) > 1 and self.replay_time is not None:
            first_time = self.replay_time[self.replay_indices[0]]
            last_time = self.replay_time[self.replay_indices[-1]]
            total_duration_s = last_time - first_time
            if total_duration_s > 0:
                avg_interval_ms = (total_duration_s * 1000) / len(self.replay_indices)
            else:
                avg_interval_ms = 40 # 25 FPS
        else:
            avg_interval_ms = 40 # Fallback to 25 FPS

        # Create the animation. blit=False is more robust for GUIs with text updates in TkAgg
        self.animation = FuncAnimation(
            self.fig,
            self._update_animation_frame,
            frames=len(self.replay_indices),
            interval=max(1, avg_interval_ms),
            blit=False, # blit=False is crucial for stability with text updates in TkAgg
            repeat=False,
            # save_count is needed to ensure the _on_animation_end callback is triggered
            save_count=len(self.replay_indices)
        )
        # Connect the end-of-animation callback
        self.animation._stop = self._on_animation_end
        self.canvas.draw()


    def stop_replay(self):
        """Pause the frame replay."""
        if self.animation is not None:
            try:
                if self.animation.event_source:
                    self.animation.event_source.stop()
            except AttributeError:
                pass
            self.animation = None

if __name__ == "__main__":
    app = ModernMiniBESTGui()
    app.mainloop()
