from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./mini_best.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    patients = relationship("Patient", back_populates="clinician")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_identifier = Column(String, index=True) # e.g. "P001"
    clinician_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    clinician = relationship("User", back_populates="patients")
    results = relationship("TestResult", back_populates="patient")

class TestResult(Base):
    __tablename__ = "test_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    test_type = Column(String) # "MINIBEST" or "FGA"
    exercise_name = Column(String) # e.g. "Sit to Stand"
    score = Column(Integer)
    details = Column(JSON) # Stores detailed metrics
    file_path = Column(String, nullable=True) # Path to raw file if saved
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="results")

def init_db():
    Base.metadata.create_all(bind=engine)

