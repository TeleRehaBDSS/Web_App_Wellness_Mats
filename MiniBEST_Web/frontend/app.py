import streamlit as st
import requests
import pandas as pd
import json
import base64

API_URL = "http://127.0.0.1:8000"

def login():
    st.title("Clinician Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            res = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
            if res.status_code == 200:
                token_data = res.json()
                st.session_state["token"] = token_data["access_token"]
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials")
        except Exception as e:
            st.error(f"Connection error: {e}")

def get_headers():
    return {"Authorization": f"Bearer {st.session_state['token']}"}

def dashboard():
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    
    if st.sidebar.button("Logout"):
        del st.session_state["token"]
        st.rerun()

    # 1. Patient Selection
    st.sidebar.header("Patient Management")
    
    # Fetch patients
    try:
        res = requests.get(f"{API_URL}/patients/", headers=get_headers())
        patients = res.json() if res.status_code == 200 else []
    except:
        patients = []
    
    patient_names = [p["patient_identifier"] for p in patients]
    selected_patient_name = st.sidebar.selectbox("Select Patient", ["Create New"] + patient_names)
    
    current_patient = None
    if selected_patient_name == "Create New":
        new_p_name = st.sidebar.text_input("New Patient ID")
        if st.sidebar.button("Create"):
            res = requests.post(f"{API_URL}/patients/", params={"patient_identifier": new_p_name}, headers=get_headers())
            if res.status_code == 200:
                st.success("Patient created!")
                st.rerun()
            else:
                st.error("Error creating patient")
    else:
        current_patient = next((p for p in patients if p["patient_identifier"] == selected_patient_name), None)

    if not current_patient:
        st.info("Please select or create a patient to continue.")
        return

    # History Expander
    with st.expander("Patient History (Saved Results)"):
        try:
            res = requests.get(f"{API_URL}/results/{current_patient['id']}", headers=get_headers())
            if res.status_code == 200:
                history = res.json()
                if history:
                    history_df = pd.DataFrame(history)
                    # Clean up display
                    display_cols = ["created_at", "test_type", "exercise_name", "score"]
                    st.dataframe(history_df[display_cols].sort_values("created_at", ascending=False))
                else:
                    st.write("No history found.")
            else:
                st.error("Failed to fetch history")
        except Exception as e:
            st.error(f"Error: {e}")

    # 2. Test Selection
    test_type = st.radio("Select Test", ["MiniBESTest", "Functional Gait Assessment (FGA)"], horizontal=True)
    
    if test_type == "MiniBESTest":
        render_minibest(current_patient)
    else:
        render_fga(current_patient)

def render_minibest(patient):
    st.header(f"MiniBESTest - Patient: {patient['patient_identifier']}")
    
    exercises = {
        "sit_to_stand": "1. Sit to Stand",
        "rise_to_toes": "2. Rise to Toes",
        "stance_eyes_open": "3. Stance (Eyes Open)",
        "compensatory_stepping_forward": "4. Compensatory Stepping - Forward",
        "compensatory_stepping_backward": "5. Compensatory Stepping - Backward",
        "compensatory_stepping_lateral": "6. Compensatory Stepping - Lateral",
        "stance_eyes_closed": "7. Stance (Eyes Closed)",
        "stand_one_leg": "8. Stand on One Leg",
        "change_gait_speed": "10. Change in Gait Speed",
        "walk_head_turns_horizontal": "11. Walk with Head Turns - Horizontal",
        "walk_pivot_turns": "12. Walk with Pivot Turns",
        "step_over_obstacles": "13. Step Over Obstacles",
        "tug_dual_task": "14. Timed Up & Go with Dual Task"
    }
    
    # Use selectbox for exercise selection (scrollable)
    selected_exercise = st.selectbox(
        "Select Exercise",
        options=list(exercises.items()),
        format_func=lambda x: x[1],
        key="minibest_exercise_selector"
    )
    
    key, label = selected_exercise
    
    # Display selected exercise
    uploaded_file = st.file_uploader(f"Upload CSV for {label}", type=["csv"], key=f"upl_{key}")
    
    # Extra inputs
    used_hands = False
    multiple_attempts = False
    variant = None
    manual_input = None
    
    col1, col2 = st.columns(2)
    if key == "sit_to_stand":
        used_hands = col1.checkbox("Used Hands?", key=f"uh_{key}")
        multiple_attempts = col2.checkbox("Multiple Attempts?", key=f"ma_{key}")
    elif key == "compensatory_stepping_lateral":
        variant = col1.selectbox("Side", ["LEFT", "RIGHT"], key=f"var_{key}")
    elif key == "stand_one_leg":
        variant = col1.selectbox("Leg", ["Left", "Right"], key=f"var_{key}")
    elif key == "tug_dual_task":
        st.info("For TUG with Dual Task, upload two CSV files: one for TUG alone, one for TUG with Dual Task")
        tug_alone_file = st.file_uploader("Upload CSV for TUG alone", type=["csv"], key=f"tug_alone_{key}")
        tug_dual_file = uploaded_file  # Use the main uploader for dual task
        uploaded_file = tug_alone_file  # Will handle both in backend

    if uploaded_file and st.button(f"Analyze {label}", key=f"btn_{key}"):
        files = {"file": uploaded_file}
        data = {
            "exercise_type": key,
            "patient_id": patient["id"],
            "used_hands": used_hands,
            "multiple_attempts": multiple_attempts,
            "variant": variant
        }
        
        # Create a more relevant loading indicator for medical analysis
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #1e1e1e; border-radius: 10px; border: 1px solid #444;">
                <h3 style="color: #4CAF50;">📊 Processing Movement Analysis</h3>
                <p style="color: #ccc;">Analyzing balance patterns and calculating metrics...</p>
                <div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
                    <div style="width: 12px; height: 12px; background: #4CAF50; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite;"></div>
                    <div style="width: 12px; height: 12px; background: #4CAF50; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite; animation-delay: 0.2s;"></div>
                    <div style="width: 12px; height: 12px; background: #4CAF50; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite; animation-delay: 0.4s;"></div>
                </div>
            </div>
            <style>
                @keyframes pulse {
                    0%, 100% { opacity: 0.3; transform: scale(0.8); }
                    50% { opacity: 1; transform: scale(1.2); }
                }
            </style>
            """, unsafe_allow_html=True)
        
        res = requests.post(f"{API_URL}/analyze/minibest", files=files, data=data, headers=get_headers())
        loading_placeholder.empty()
                
        if res.status_code == 200:
            result = res.json()
            st.success(f"Score: {result['score']}")
            
            # Layout handling based on exercise type
            if key == "sit_to_stand":
                st.markdown("### Analysis Dashboard")
                row1 = st.columns(2)
                row2 = st.columns(2)
                
                # 1. Video (Top Left)
                with row1[0]:
                    if result.get("replay"):
                        st.markdown("##### Video Replay")
                        st.image(base64.b64decode(result["replay"]), width=400)
                    else:
                        st.info("Video not available")
                        
                # 2. Force Plot (Top Right)
                with row1[1]:
                    if result.get("force_plot"):
                        st.markdown("##### Force Profile")
                        st.image(base64.b64decode(result["force_plot"]), use_container_width=True)
                        
                # 3. AP Plot (Bottom Left)
                with row2[0]:
                    if result.get("ap_plot"):
                        st.markdown("##### AP Displacement")
                        st.image(base64.b64decode(result["ap_plot"]), use_container_width=True)
                        
                # 4. Stability Plot (Bottom Right)
                with row2[1]:
                    if result.get("stability_plot"):
                        st.markdown("##### Stability Map")
                        st.image(base64.b64decode(result["stability_plot"]), use_container_width=True)

            elif key == "rise_to_toes":
                st.markdown("### Analysis Dashboard")
                row1 = st.columns(2)
                row2 = st.columns(2)
                
                # 1. Video (Top Left)
                with row1[0]:
                    if result.get("replay"):
                        st.markdown("##### Video Replay")
                        st.image(base64.b64decode(result["replay"]), width=400)
                    else:
                        st.info("Video not available")
                        
                # 2. Area Plot (Top Right)
                with row1[1]:
                    if result.get("force_plot"): # Reusing key for Contact Area
                        st.markdown("##### Contact Area (Heel Rise)")
                        st.image(base64.b64decode(result["force_plot"]), use_container_width=True)
                        
                # 3. AP Plot (Bottom Left)
                with row2[0]:
                    if result.get("ap_plot"):
                        st.markdown("##### Anterior-Posterior Balance")
                        st.image(base64.b64decode(result["ap_plot"]), use_container_width=True)
                        
                # 4. Stability Plot (Bottom Right)
                with row2[1]:
                    if result.get("stability_plot"):
                        st.markdown("##### CoP Stability (Flat -> Toes)")
                        st.image(base64.b64decode(result["stability_plot"]), use_container_width=True)
            
            elif key in ["compensatory_stepping_forward", "compensatory_stepping_backward", "compensatory_stepping_lateral"]:
                # Compensatory stepping exercises
                st.markdown("### Analysis Dashboard")
                row1 = st.columns(2)
                row2 = st.columns(2)
                
                # 1. Video (Top Left)
                with row1[0]:
                    if result.get("replay"):
                        st.markdown("##### Video Replay")
                        st.image(base64.b64decode(result["replay"]), width=400)
                    else:
                        st.info("Video not available")
                        
                # 2. CoP Trajectory (Top Right)
                with row1[1]:
                    if result.get("force_plot"):  # Reused for CoP trajectory
                        st.markdown("##### CoP Trajectory Over Time")
                        st.image(base64.b64decode(result["force_plot"]), use_container_width=True)
                        
                # 3. Stabilogram (Bottom Left)
                with row2[0]:
                    if result.get("ap_plot"):  # Reused for Stabilogram
                        st.markdown("##### CoP Stabilogram (X vs Y)")
                        st.image(base64.b64decode(result["ap_plot"]), use_container_width=True)
                        
                # 4. Step Analysis (Bottom Right)
                with row2[1]:
                    if result.get("stability_plot"):
                        st.markdown("##### Step Analysis")
                        st.image(base64.b64decode(result["stability_plot"]), use_container_width=True)
            
            elif key in ["stance_eyes_open", "stance_eyes_closed", "stand_one_leg"]:
                st.markdown("### Analysis Dashboard")
                row1 = st.columns(2)
                row2 = st.columns(2)
                
                # 1. Video (Top Left)
                with row1[0]:
                    if result.get("replay"):
                        st.markdown("##### Video Replay")
                        st.image(base64.b64decode(result["replay"]), width=400)
                    else:
                        st.info("Video not available")
                        
                # 2. Sway Plot (Top Right)
                with row1[1]:
                    if result.get("force_plot"): # Reused for Sway Time
                        st.markdown("##### CoP Sway Over Time")
                        st.image(base64.b64decode(result["force_plot"]), use_container_width=True)
                        
                # 3. Stabilogram (Bottom Left)
                with row2[0]:
                    if result.get("ap_plot"): # Reused for Stabilogram
                        st.markdown("##### CoP Stabilogram (X vs Y)")
                        st.image(base64.b64decode(result["ap_plot"]), use_container_width=True)
                        
                # 4. Stability Map (Bottom Right)
                with row2[1]:
                    if result.get("stability_plot"):
                        st.markdown("##### CoP Stability Map")
                        st.image(base64.b64decode(result["stability_plot"]), use_container_width=True)

            elif key in ["change_gait_speed", "walk_head_turns_horizontal", "walk_pivot_turns", "step_over_obstacles", "tug_dual_task"]:
                # Gait exercises - similar layout to FGA
                st.markdown("### Analysis Dashboard")
                
                # Video Replay
                if result.get("replay"):
                    st.subheader("Gait Video Replay")
                    st.image(base64.b64decode(result["replay"]), use_container_width=True, caption="Gait across mats over time")
                
                # Gait Analysis Plots
                if result.get("stride_plot") or result.get("cadence_plot") or result.get("cop_trajectories_plot") or result.get("butterfly_cop_plot"):
                    st.subheader("Gait Analysis Plots")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if result.get("stride_plot"):
                            st.image(base64.b64decode(result["stride_plot"]), use_container_width=True, caption="Stride Length Over Time")
                        if result.get("cadence_plot"):
                            st.image(base64.b64decode(result["cadence_plot"]), use_container_width=True, caption="Cadence Over Time")
                    with col2:
                        if result.get("cop_trajectories_plot"):
                            st.image(base64.b64decode(result["cop_trajectories_plot"]), use_container_width=True, caption="CoP Trajectories")
                        if result.get("butterfly_cop_plot"):
                            st.image(base64.b64decode(result["butterfly_cop_plot"]), use_container_width=True, caption="Butterfly CoP Plot")
                
                # Metrics
                with st.expander("Detailed Metrics"):
                    st.json(result.get('features', {}))
            else:
                # Fallback for other exercises
                if result.get("plot"):
                    st.image(base64.b64decode(result["plot"]), caption="Analysis Plots", use_container_width=True)
                if result.get("replay"):
                    st.image(base64.b64decode(result["replay"]), caption="Video Replay", width=400)
                
            if key not in ["change_gait_speed", "walk_head_turns_horizontal", "walk_pivot_turns", "step_over_obstacles", "tug_dual_task"]:
                st.json(result['features'])
        else:
            st.error(f"Error: {res.text}")

def render_fga(patient):
    st.header(f"FGA - Patient: {patient['patient_identifier']}")
    
    fga_exercises = {
        1: "Gait Level Surface",
        2: "Change in Gait Speed",
        3: "Gait with Horizontal Head Turns",
        4: "Gait with Vertical Head Turns",
        5: "Gait and Pivot Turn",
        6: "Step Over Obstacle",
        7: "Gait with Narrow Base of Support",
        8: "Gait with Eyes Closed",
        9: "Ambulating Backwards",
        10: "Steps"
    }
    
    # Use selectbox for exercise selection (scrollable)
    selected_exercise = st.selectbox(
        "Select Exercise",
        options=list(fga_exercises.items()),
        format_func=lambda x: f"{x[0]}. {x[1]}",
        key="fga_exercise_selector"
    )
    
    ex_num, label = selected_exercise
    
    # Display selected exercise
    uploaded_file = st.file_uploader(f"Upload CSV for {label}", type=["csv"], key=f"fga_upl_{ex_num}")
    
    manual_input = None
    if ex_num in [3, 4, 5, 6]:
        options = {
            3: ["Smoothly", "Mild difficulty", "Moderate difficulty", "Severe difficulty or unable"],
            4: ["Smoothly", "Mild difficulty", "Moderate difficulty", "Severe difficulty or unable"],
            5: ["Smooth and balanced", "Mild imbalance", "Significant imbalance or hesitation", "Unable to perform"],
            6: ["Smooth", "Slight hesitation", "Significant effort or imbalance", "Unable to perform"]
        }
        manual_input = st.selectbox("Manual Rating (Clinical Observation)", options[ex_num], key=f"man_{ex_num}")

    if uploaded_file and st.button(f"Analyze {label}", key=f"fga_btn_{ex_num}"):
        files = {"file": uploaded_file}
        data = {
            "exercise_num": ex_num,
            "patient_id": patient["id"],
            "manual_input": manual_input
        }
        
        # Create a more relevant loading indicator for gait analysis
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div style="text-align: center; padding: 20px; background-color: #1e1e1e; border-radius: 10px; border: 1px solid #444;">
                <h3 style="color: #2196F3;">🚶 Analyzing Gait Patterns</h3>
                <p style="color: #ccc;">Processing pressure mat data and calculating gait metrics...</p>
                <div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
                    <div style="width: 12px; height: 12px; background: #2196F3; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite;"></div>
                    <div style="width: 12px; height: 12px; background: #2196F3; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite; animation-delay: 0.2s;"></div>
                    <div style="width: 12px; height: 12px; background: #2196F3; border-radius: 50%; animation: pulse 1.5s ease-in-out infinite; animation-delay: 0.4s;"></div>
                </div>
            </div>
            <style>
                @keyframes pulse {
                    0%, 100% { opacity: 0.3; transform: scale(0.8); }
                    50% { opacity: 1; transform: scale(1.2); }
                }
            </style>
            """, unsafe_allow_html=True)
        
        res = requests.post(f"{API_URL}/analyze/fga", files=files, data=data, headers=get_headers())
        loading_placeholder.empty()
        
        if res.status_code == 200:
            result = res.json()
            
            # Display Score
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("FGA Score", result['score'], help="Score: 0=Severe, 1=Moderate, 2=Mild, 3=Normal")
            with col2:
                st.info(f"**Explanation:** {result['explanation']}")
            
            # Display Video Replay (for all exercises)
            if result.get("replay"):
                st.subheader("Gait Video Replay (Treadmill View)")
                st.image(base64.b64decode(result["replay"]), use_container_width=True, caption="Gait across 6 mats over time")
            
            # Display Gait Analysis Plots (for all exercises)
            if result.get("stride_plot") or result.get("cadence_plot") or result.get("cop_trajectories_plot") or result.get("butterfly_cop_plot"):
                st.subheader("Gait Analysis Plots")
                
                # Stride Plot
                if result.get("stride_plot"):
                    st.image(base64.b64decode(result["stride_plot"]), use_container_width=True, caption="Stride Length Over Time")
                
                # Cadence Plot
                if result.get("cadence_plot"):
                    st.image(base64.b64decode(result["cadence_plot"]), use_container_width=True, caption="Cadence Over Time")
                
                # CoP Trajectories Plot
                if result.get("cop_trajectories_plot"):
                    st.image(base64.b64decode(result["cop_trajectories_plot"]), use_container_width=True, caption="CoP Trajectories (Left vs Right Foot)")
                
                # Butterfly CoP Plot (for applicable exercises)
                if result.get("butterfly_cop_plot"):
                    st.image(base64.b64decode(result["butterfly_cop_plot"]), use_container_width=True, caption="Butterfly CoP Plot (Continuous CoP Trace)")
            
            # Display Metrics
            with st.expander("Detailed Metrics"):
                st.json(result.get('features', result.get('metrics', {})))
        else:
            st.error(f"Error: {res.text}")

def main():
    st.set_page_config(
        page_title="MiniBEST & FGA Dashboard", 
        layout="wide",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "MiniBEST & FGA Analysis Dashboard - Balance Assessment Tool"
        }
    )
    
    # Hide only the menu icon (hamburger menu) in the top right, not the sidebar
    st.markdown("""
    <style>
        /* Hide the hamburger menu button */
        button[title="View app source"] {display: none;}
        button[data-testid="baseButton-header"] {display: none;}
        /* Hide the "Deploy" button */
        .stDeployButton {display: none;}
        /* Hide footer */
        footer {visibility: hidden;}
        /* Keep sidebar visible - don't hide header completely */
        [data-testid="stSidebar"] {visibility: visible !important;}
    </style>
    """, unsafe_allow_html=True)
    
    if "token" not in st.session_state:
        login()
    else:
        dashboard()

if __name__ == "__main__":
    main()
