# MiniBEST & FGA Analysis Web App

## Overview
This is a full-stack web application for clinicians to analyze balance tests (MiniBESTest and FGA) using pressure mat data.

### Features
- **Clinician Login**: Secure access for 5 distinct users.
- **Patient Management**: Create and select patients.
- **Test Selection**: Switch between MiniBESTest and FGA batteries.
- **Data Analysis**: Upload CSV files from pressure mats to automatically calculate scores and metrics.
- **Results Dashboard**: View scores, explanations, and detailed metrics.

## Project Structure
- `backend/`: FastAPI application handling authentication, database, and analysis logic.
- `frontend/`: Streamlit dashboard for user interaction.
- `data/`: Storage for SQLite database (`mini_best.db`) and uploaded files.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

You need to run the Backend and Frontend in separate terminals.

### 1. Start the Backend (API)
```bash
uvicorn backend.main:app --reload
```
*The API will be available at http://127.0.0.1:8000*

### 2. Start the Frontend (Dashboard)
```bash
streamlit run frontend/app.py
```
*The Dashboard will open in your browser at http://localhost:8501*

## Login Credentials
Default users created on first run:
- **Username**: clinician1 (up to clinician5)
- **Password**: password123

## Usage
1. Log in as a clinician.
2. Create a new Patient (or select an existing one).
3. Choose the Test Battery (MiniBESTest or FGA).
4. Select the specific exercise tab.
5. Upload the CSV file corresponding to that exercise.
6. Provide any required manual inputs (checkboxes or dropdowns).
7. Click "Analyze" to see the results.

