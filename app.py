import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import base64
import os
import json
import requests
import dash
from dash import dcc, html, Input, Output, State
import base64
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import MATCH, ALL


# Initialize Dash app
app = dash.Dash(__name__)

# Flask API URL
API_URL = "http://127.0.0.1:5000/upload"

# Create upload directory
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# FGA items with manual inputs and automatic metrics
fga_items = [
    {
        "id": "1",
        "name": "Gait Level Surface",
        "automatic_metrics": [
            "Time to walk 6m",
            "Deviation from a straight line",
            "Cadence",
            "Step length",
            "Stride length",
            "Single/double support times",
            "Smoothness",
            "Shuffling gait",
            "Uneven steps",
            "Hesitant movement"
        ],
        "manual_inputs": []
    },
    {
        "id": "2",
        "name": "Change in Gait Speed",
        "automatic_metrics": [
            "Average speed for normal, fast, and slow walking",
            "Smoothness of transition",
            "Gait cycle consistency during speed changes"
        ],
        "manual_inputs": []
    },
    {
        "id": "3",
        "name": "Gait with Horizontal Head Turns",
        "automatic_metrics": [
            "Gait speed and deviation during head turns",
            "Time to complete 6m walk",
            "Stability assessment during turns"
        ],
        "manual_inputs": [
            {
                "id": "smoothness_horizontal",
                "label": "Smoothness of head turns",
                "options": ["Smoothly", "Mild difficulty", "Moderate difficulty", "Severe difficulty or unable"]
            }
        ]
    },
    {
        "id": "4",
        "name": "Gait with Vertical Head Turns",
        "automatic_metrics": [
            "Gait speed and deviation during head turns",
            "Time to complete 6m walk",
            "Stability assessment during turns"
        ],
        "manual_inputs": [
            {
                "id": "smoothness_vertical",
                "label": "Smoothness of vertical head turns",
                "options": ["Smoothly", "Mild difficulty", "Moderate difficulty", "Severe difficulty or unable"]
            }
        ]
    },
    {
        "id": "5",
        "name": "Gait and Pivot Turn",
        "automatic_metrics": [
            "Time to turn 180°",
            "Balance during pivot",
            "Deviation from a stable stance post-turn"
        ],
        "manual_inputs": [
            {
                "id": "smoothness_pivot",
                "label": "Smoothness of pivot turn",
                "options": ["Smooth and balanced", "Mild imbalance", "Significant imbalance or hesitation", "Unable to pivot"]
            }
        ]
    },
    {
        "id": "6",
        "name": "Step Over Obstacle",
        "automatic_metrics": [
            "Gait speed before, during, and after the obstacle",
            "Deviations and hesitations while stepping over",
            "Step height (if supported)"
        ],
        "manual_inputs": [
            {
                "id": "smoothness_obstacle",
                "label": "Smoothness of obstacle negotiation",
                "options": ["Smooth", "Slight hesitation", "Significant effort or imbalance", "Unable to perform"]
            }
        ]
    },
    {
        "id": "7",
        "name": "Gait with Narrow Base of Support",
        "automatic_metrics": [
            "Number of tandem steps",
            "Deviation from a straight line",
            "Balance stability"
        ],
        "manual_inputs": []
    },
    {
        "id": "8",
        "name": "Gait with Eyes Closed",
        "automatic_metrics": [
            "Time to walk 6m",
            "Deviation in walking pattern",
            "Stability during walking"
        ],
        "manual_inputs": []
    },
    {
        "id": "9",
        "name": "Ambulating Backwards",
        "automatic_metrics": [
            "Speed and deviation during backward walking",
            "Gait cycle parameters (cadence, step length, stride length)",
            "Balance stability during reverse steps"
        ],
        "manual_inputs": []
    },
    {
        "id": "10",
        "name": "Steps",
        "automatic_metrics": [],  # No automatic metrics
        "manual_inputs": [
            {
                "id": "smoothness",
                "label": "Smoothness of stepping",
                "options": ["Smooth and balanced", "Mild difficulty", "Significant imbalance", "Unable to perform"],
                "placeholder": "Select smoothness level",
            },
            {
                "id": "effort",
                "label": "Perceived effort",
                "options": ["Low", "Moderate", "High", "Unable to perform"],
                "placeholder": "Select effort level",
            },
            {
                "id": "balance",
                "label": "Balance during ascent/descent",
                "options": ["Stable", "Slightly unstable", "Moderately unstable", "Severely unstable"],
                "placeholder": "Select balance stability",
            },
            {
                "id": "fatigue",
                "label": "Fatigue level (0–10)",
                "options": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                "placeholder": "Rate fatigue level",
            }

        ]
    }
]


# Layout for each FGA item tab
def generate_item_tab(item):
    manual_inputs = []

    # Always include at least one manual input
    if item["manual_inputs"]:
        for manual_input in item["manual_inputs"]:
            manual_inputs.append(html.Label(manual_input["label"]))
            manual_inputs.append(dcc.Dropdown(
                id={"type": "manual-input", "id": item["id"], "field": manual_input.get("id", "default")},
                options=[{"label": opt, "value": opt} for opt in manual_input["options"]],
                placeholder=manual_input.get("placeholder", "Select an option"),
            ))
    else:
        # Add a default manual input for exercises with no manual inputs
        manual_inputs.append(html.Label("No manual input required"))
        manual_inputs.append(dcc.Dropdown(
            id={"type": "manual-input", "id": item["id"], "field": "default"},
            options=[],  # Empty options
            placeholder="No manual input required",
            disabled=True,  # Disable the dropdown
        ))

    return dcc.Tab(
        label=f"{item['id']}. {item['name']}",
        value=item['id'],  # Ensure value is a string
        children=html.Div([
            html.H3(f"{item['name']}: Automatic and Manual Inputs"),
            dcc.Upload(
                id={"type": "file_upload", "id": item["id"]},
                children=html.Div(["Drag and Drop or ", html.A("Select a File")]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                multiple=False,
            ),
            html.Div(id={"type": "upload-feedback", "id": item["id"]}, style={"marginTop": "10px", "color": "green"}),
            html.H4("Manual Inputs:"),
            html.Div(manual_inputs),
            html.Button("Calculate", id={"type": "calculate-button", "id": item["id"]}, style={"marginTop": "10px"}),

            # Add section for grade and explanation
            html.Div(id={"type": "grade-feedback", "id": item["id"]}, style={"marginTop": "10px", "color": "black"}),

            html.Div(id={"type": "results", "id": item["id"]}, style={"marginTop": "10px", "color": "blue"}),
        ])
    )





# Dash Layout
app.layout = html.Div([
    html.H1("FGA Assessment Dashboard"),
    dcc.Tabs(
        id="fga-tabs",
        value=fga_items[0]['id'],  # Ensure value is a string
        children=[generate_item_tab(item) for item in fga_items],
    )
])

# Callbacks for file upload and metric computation
@app.callback(
    [
        Output({"type": "upload-feedback", "id": MATCH}, "children"),
        Output({"type": "results", "id": MATCH}, "children"),
        Output({"type": "grade-feedback", "id": MATCH}, "children"),
    ],
    [
        Input({"type": "file_upload", "id": MATCH}, "contents"),
        Input({"type": "calculate-button", "id": MATCH}, "n_clicks"),
    ],
    [
        State({"type": "file_upload", "id": MATCH}, "filename"),
        State({"type": "file_upload", "id": MATCH}, "id"),
        State({"type": "manual-input", "id": MATCH, "field": ALL}, "value"),
    ],
)
def handle_file_upload(contents, n_clicks, filename, exercise_id, manual_inputs):
    print(f"Callback triggered: n_clicks={n_clicks}, contents={contents}")
    print(f"Filename: {filename}, Exercise ID: {exercise_id}")
    print(f"Manual Inputs: {manual_inputs}")

    # Check if manual inputs are required
    if exercise_id["id"] in ["1", "2", "7", "8", "9"]:  # Exercises without manual inputs
        manual_inputs = None

    # Handle missing manual inputs
    if manual_inputs is None or (isinstance(manual_inputs, list) and all(input_value is None for input_value in manual_inputs)):
        if exercise_id["id"] not in ["1", "2", "7", "8", "9"]:  # Exercises requiring manual inputs
            return "Please provide all required manual inputs.", html.Div(), html.Div()

    if contents:
        try:
            print('exercise_id = ', exercise_id)
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Save the file temporarily
            upload_path = os.path.join(UPLOAD_DIRECTORY, filename)
            with open(upload_path, "wb") as f:
                f.write(decoded)

            # Send the file to the backend API
            response = requests.post(
                API_URL,
                files={"file": (filename, decoded)}
            )
            print(contents)
            if response.status_code == 200:
                
                metrics = response.json()["metrics"]
                feedback = f"File '{filename}' uploaded successfully."

                # Generate the grade and explanation
                if exercise_id['id'] == "1":  # For "Gait Level Surface"
                    from backend.metrics import grade_gait_level_surface
                    grade, explanation = grade_gait_level_surface(metrics)
                    #grading_result = f"Grade: {grade}\nExplanation: {explanation}"

                elif exercise_id["id"] == "2":  # "Change in Gait Speed"
                    from backend.metrics import exercise02
                    grade, explanation = exercise02(metrics)

                elif exercise_id["id"] == "3":  # "Gait with Horizontal Head Turns"
                    if manual_inputs:
                        from backend.metrics import exercise03
                        grade, explanation = exercise03(metrics, manual_inputs[0])
                    else:
                        explanation = "Manual input is required for grading this exercise."

                elif exercise_id["id"] == "4":  # "Gait with Vertical Head Turns"
                    if manual_inputs:
                        from backend.metrics import exercise04
                        grade, explanation = exercise04(metrics, manual_inputs[0])
                    else:
                        explanation = "Manual input is required for grading this exercise."

                elif exercise_id["id"] == "5":  # "Gait and Pivot Turn"
                    if manual_inputs:
                        from backend.metrics import exercise05
                        grade, explanation = exercise05(metrics, manual_inputs[0])
                    else:
                        explanation = "Manual input is required for grading this exercise."

                elif exercise_id["id"] == "6":  # "Step Over Obstacle"
                    if manual_inputs:
                        from backend.metrics import exercise06
                        grade, explanation = exercise06(metrics, manual_inputs[0])
                    else:
                        explanation = "Manual input is required for grading this exercise."

                elif exercise_id["id"] == "7":  # "Gait with Narrow Base of Support"
                    from backend.metrics import exercise07
                    grade, explanation = exercise07(metrics)

                elif exercise_id["id"] == "8":  # "Gait with Eyes Closed"
                    from backend.metrics import exercise08
                    grade, explanation = exercise08(metrics)

                elif exercise_id["id"] == "9":  # "Ambulating Backwards"
                    from backend.metrics import exercise09
                    grade, explanation = exercise09(metrics)

                elif exercise_id["id"] == "10":  # Steps
                    from backend.metrics import exercise10
                    # Convert manual_inputs to a dictionary with required keys
                    manual_inputs_dict = {input["field"]: input["value"] for input in manual_inputs}
                    grade, explanation = exercise10(manual_inputs_dict)



                # Grading result HTML
                grading_result_html = html.Div([
                    html.H4("Grading Result"),
                    html.Div([
                        html.P(f"Grade: {grade}", style={"fontWeight": "bold", "color": "green"}),
                        html.P(f"Explanation: {explanation}"),
                    ], style={
                        "marginBottom": "20px",
                        "padding": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "5px",
                        "backgroundColor": "#f9f9f9",
                    }),
                ])
                if exercise_id["id"] != "10":

                    # Generate graphs (same as your current implementation)
                    overview_graph = dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Bar(name="Average Cadence", x=["Average Cadence"], y=[metrics["average_cadence"]]),
                                go.Bar(name="Number of Steps", x=["Number of Steps"], y=[metrics["number_of_steps"]]),
                            ],
                            layout_title_text="Overview of Metrics"
                        )
                    )
                    gait_cycles = metrics["gait_cycles"]
                    gait_cycles_df = pd.DataFrame(gait_cycles)
                    gait_cycles_graph = dcc.Graph(
                        figure=px.line(
                            gait_cycles_df,
                            x="gait_cycle_id",
                            y=["cadence", "double_support_time", "single_support_time"],
                            title="Gait Cycle Metrics"
                        )
                    )
                    distances_graph = dcc.Graph(
                        figure=px.scatter(
                            gait_cycles_df,
                            x="gait_cycle_id",
                            y=["average_horizontal_distance", "average_vertical_distance"],
                            title="Gait Cycle Distances"
                        )
                    )
                    support_times_graph = dcc.Graph(
                        figure=px.bar(
                            gait_cycles_df,
                            x="gait_cycle_id",
                            y=["double_support_time", "single_support_time"],
                            title="Support Times per Gait Cycle",
                            barmode="group"
                        )
                    )
                    results = html.Div([
                        html.Div(overview_graph, style={"marginBottom": "20px"}),
                        html.H3("Gait Cycles"),
                        gait_cycles_graph,
                        html.H3("Distances"),
                        distances_graph,
                        html.H3("Support Times"),
                        support_times_graph,
                    ])
                else:
                    feedback = f"Error uploading '{filename}': {response.text}"
                    results = html.Div()
                    grading_result_html = html.Div()

        except Exception as e:
            feedback = f"An error occurred with '{filename}': {str(e)}"
            results = html.Div()
            grading_result_html = html.Div()

        return feedback, results, grading_result_html
    
    elif exercise_id["id"] == "10":  # Steps
        from backend.metrics import exercise10
        # Convert manual_inputs to a dictionary with required keys
        print('manual_inputs = ', manual_inputs)

        manual_input_keys = ["smoothness", "effort", "balance", "fatigue"]

        # Map values to keys
        if len(manual_inputs) == len(manual_input_keys):
            manual_inputs_dict = dict(zip(manual_input_keys, manual_inputs))
        else:
            manual_inputs_dict = {}
        #manual_inputs_dict = {input["field"]: int(input["value"]) for input in manual_inputs}
        grade, explanation = exercise10(manual_inputs_dict)
        feedback = " "

        # Grading result HTML
        grading_result_html = html.Div([
            html.H4("Grading Result"),
            html.Div([
                html.P(f"Grade: {grade}", style={"fontWeight": "bold", "color": "green"}),
                html.P(f"Explanation: {explanation}"),
            ], style={
                "marginBottom": "20px",
                "padding": "10px",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
                "backgroundColor": "#f9f9f9",
            }),
        ])
        results = html.Div()
        #grading_result_html = html.Div()
        return feedback, results, grading_result_html

        

    return "No file uploaded yet.", html.Div(), ""






if __name__ == "__main__":
    app.run_server(debug=True)
