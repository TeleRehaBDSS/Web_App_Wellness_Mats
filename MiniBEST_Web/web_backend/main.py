import os
import shutil
import base64
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
import math
import numpy as np

from .database import engine, SessionLocal, init_db, User, Patient, TestResult
from .auth import (
    get_current_user, create_access_token, get_password_hash, 
    verify_password, get_db, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .analysis import minibest, metrics as fga_metrics
from .analysis.plotting_utils import generate_plot_components, generate_video

# Initialize DB
init_db()

app = FastAPI(title="MiniBEST & FGA Analysis API")

# Helper for JSON compliance
def sanitize_float(val):
    """
    Recursively replace NaN/Infinity with None (null in JSON).
    """
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(val, dict):
        return {k: sanitize_float(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_float(v) for v in val]
    elif isinstance(val, np.ndarray):
        # Convert numpy arrays to lists and then sanitize
        return sanitize_float(val.tolist())
    elif isinstance(val, (np.float32, np.float64)):
        # Handle numpy scalar floats
        return sanitize_float(float(val))
    return val

# Create initial users if not exist
def create_initial_users():
    db = SessionLocal()
    users = ["clinician1", "clinician2", "clinician3", "clinician4", "clinician5"]
    for u in users:
        if not db.query(User).filter(User.username == u).first():
            user = User(username=u, hashed_password=get_password_hash("password123"))
            db.add(user)
    db.commit()
    db.close()

create_initial_users()

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "id": current_user.id}

# Patient Endpoints

@app.post("/patients/")
async def create_patient(patient_identifier: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Check if exists for this clinician
    existing = db.query(Patient).filter(Patient.patient_identifier == patient_identifier, Patient.clinician_id == current_user.id).first()
    if existing:
        return existing
    
    patient = Patient(patient_identifier=patient_identifier, clinician_id=current_user.id)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

@app.get("/patients/")
async def get_patients(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Patient).filter(Patient.clinician_id == current_user.id).all()

# Analysis Endpoints

@app.post("/analyze/minibest")
async def analyze_minibest(
    file: UploadFile = File(...), 
    exercise_type: str = Form(...), 
    variant: Optional[str] = Form(None),
    patient_id: int = Form(...),
    used_hands: bool = Form(False),
    multiple_attempts: bool = Form(False),
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    # Verify patient belongs to clinician
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.clinician_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Save file
    upload_dir = f"data/uploads/{current_user.id}/{patient_id}"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = f"{upload_dir}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run Analysis
    try:
        signals = minibest.load_basic_signals(file_path)
        result = None
        
        if exercise_type == "sit_to_stand":
            result = minibest.process_sit_to_stand(signals, patient.patient_identifier, used_hands, multiple_attempts)
        elif exercise_type == "rise_to_toes":
            result = minibest.process_rise_to_toes(signals, patient.patient_identifier)
        elif exercise_type == "stance_eyes_open":
            result = minibest.process_stance_eyes_open(signals, patient.patient_identifier)
        elif exercise_type == "stance_eyes_closed":
            result = minibest.process_stance_eyes_closed(signals, patient.patient_identifier)
        elif exercise_type == "compensatory_stepping":
            result = minibest.process_compensatory_stepping(signals, variant or "FORWARD", patient.patient_identifier)
        elif exercise_type == "stand_one_leg":
            result = minibest.process_stand_on_one_leg(signals, patient.patient_identifier, variant or "Left")
        else:
            raise HTTPException(status_code=400, detail="Unknown exercise type")
            
        # Sanitize results before saving/returning
        sanitized_features = sanitize_float(result.features)
        
        # Generate Plots Components
        # Adjust exercise type for plotting if needed (e.g. stand_one_leg needs side)
        plot_type = exercise_type
        if exercise_type == "stand_one_leg":
            v_str = variant.lower() if variant else "left"
            plot_type = f"stand_one_leg_{v_str}"
            
        plot_data = generate_plot_components(plot_type, signals, sanitized_features)
        
        # Generate Video (Replay)
        video_b64 = None
        if exercise_type in ["sit_to_stand", "stance_eyes_open", "stance_eyes_closed", "rise_to_toes", "stand_one_leg"]:
             video_b64 = generate_video(signals, fps=10)
             if video_b64:
                 plot_data["replay"] = video_b64

        # Save Result Files to Disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"data/results/{current_user.id}/{patient_id}/{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        for key, b64_data in plot_data.items():
            if b64_data:
                try:
                    ext = "gif" if key == "replay" else "png"
                    file_name = f"{key}.{ext}"
                    with open(f"{result_dir}/{file_name}", "wb") as f:
                        f.write(base64.b64decode(b64_data))
                    sanitized_features[f"saved_{key}_path"] = f"{result_dir}/{file_name}"
                except Exception as e:
                    print(f"Error saving {key}: {e}")

        # Save Result to DB
        db_result = TestResult(
            patient_id=patient_id,
            test_type="MINIBEST",
            exercise_name=exercise_type,
            score=result.score,
            details=sanitized_features,
            file_path=file_path
        )
        db.add(db_result)
        db.commit()
        
        # Return everything including the individual plots
        response_data = {
            "score": result.score, 
            "features": sanitized_features
        }
        response_data.update(plot_data) # Merge plots into response
        
        return response_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/fga")
async def analyze_fga(
    file: UploadFile = File(...), 
    exercise_num: int = Form(...), # 1-10
    patient_id: int = Form(...),
    manual_input: Optional[str] = Form(None),
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.clinician_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    upload_dir = f"data/uploads/{current_user.id}/{patient_id}"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = f"{upload_dir}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Calculate base metrics
        metrics = fga_metrics.calculate_metrics(file_path)
        if "error" in metrics:
             raise Exception(metrics["error"])
             
        # Grade specific exercise
        score = 0
        explanation = ""
        
        if exercise_num == 1: # Gait Level Surface
            score, explanation = fga_metrics.grade_gait_level_surface(metrics)
        elif exercise_num == 2: # Change Speed
            score, explanation = fga_metrics.exercise02(metrics)
        elif exercise_num == 3: # Horizontal Head Turns
            score, explanation = fga_metrics.exercise03(metrics, manual_input)
        elif exercise_num == 4: # Vertical Head Turns
            score, explanation = fga_metrics.exercise04(metrics, manual_input)
        elif exercise_num == 5: # Pivot Turn
            score, explanation = fga_metrics.exercise05(metrics, manual_input)
        elif exercise_num == 6: # Step Over Obstacle
            score, explanation = fga_metrics.exercise06(metrics, manual_input)
        elif exercise_num == 7: # Narrow Base
            score, explanation = fga_metrics.exercise07(metrics)
        elif exercise_num == 8: # Eyes Closed
            score, explanation = fga_metrics.exercise08(metrics)
        elif exercise_num == 9: # Backward
            score, explanation = fga_metrics.exercise09(metrics)
        elif exercise_num == 10: # Steps
            # Exercise 10 is manual inputs only?
            # The logic provided in exercise10 accepts a dict
             score, explanation = fga_metrics.exercise10({"smoothness": manual_input, "effort": "Low", "balance": "Stable", "fatigue": 0}) # Simplification
             
        # Remove large arrays from metrics before saving to DB (JSON limits)
        # We keep summary stats.
        summary_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (list, np.ndarray))}
        summary_metrics["explanation"] = explanation
        
        sanitized_metrics = sanitize_float(summary_metrics)
        
        db_result = TestResult(
            patient_id=patient_id,
            test_type="FGA",
            exercise_name=f"Exercise {exercise_num}",
            score=score,
            details=sanitized_metrics,
            file_path=file_path
        )
        db.add(db_result)
        db.commit()
        
        return {"score": score, "explanation": explanation, "metrics": sanitized_metrics}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel

class TestResultSchema(BaseModel):
    id: int
    test_type: str
    exercise_name: str
    score: int
    created_at: datetime
    
    class Config:
        orm_mode = True

@app.get("/results/{patient_id}", response_model=List[TestResultSchema])
async def get_results(patient_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.clinician_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return db.query(TestResult).filter(TestResult.patient_id == patient_id).all()
