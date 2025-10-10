
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "postgresql://postgres:root@localhost:5432/fruad_apidb"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True, index=True)
    Month = Column(String)
    DayOfWeek = Column(String)
    Make = Column(String)
    AccidentArea = Column(String)
    Sex = Column(String)
    MaritalStatus = Column(String)
    Age = Column(Integer)
    PolicyType = Column(String)
    VehicleCategory = Column(String)
    VehiclePrice = Column(String)
    NumberOfCars = Column(Integer)
    submitted_at = Column(DateTime, default=datetime.datetime.utcnow)

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"))
    fraud_probability = Column(Float)
    risk_level = Column(String)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)
