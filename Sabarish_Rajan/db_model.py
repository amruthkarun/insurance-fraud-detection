from sqlalchemy import create_engine, Column, Integer, String, Float, DateTIme, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


DB_URL = 'postgresql://postgres:1905@localhost:5432/claims'

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InsuranceClaim(Base):
    __tablename__ = 'insurance_claims'

    months_as_customer = Column(Integer)
    age = Column(Integer)
    policy_number = Column(String(50))
    policy_bind_date = Column(DateTime)
    policy_state = Column(String(10))
    policy_csl = Column(String(10))
    policy_deductable = Column(Integer)
    policy_annual_premium = Column(Float)
    umbrella_limit = Column(Integer)
    insured_zip = Column(String(10))
    insured_sex = Column(String(10))
    insured_education_level = Column(String(10))
    insured_occupation = Column(String(10))
    insured_hobbies = Column(String(10))
    insured_relationship = Column(String(10))
    capital_gains = Column(Integer) 
    capital_loss = Column(Integer)
    incident_date = Column(DateTime)
    incident_type = Column(String(10))
    collision_type = Column(String(10))
    incident_severity = Column(String(10))
    authorities_contacted = Column(String(10))
    incident_state = Column(String(10))
    incident_city = Column(String(10))
    incident_location = Column(String(10))
    incident_hour_of_the_day = Column(Integer)
    number_of_vehicles_involved = Column(Integer)
    property_damage = Column(String(10))
    bodily_injuries = Column(Integer)
    witnesses = Column(Integer)
    police_report_available = Column(String(10))
    total_claim_amount = Column(Integer)
    injury_claim = Column(Integer)
    property_claim = Column(Integer)
    vehicle_claim = Column(Integer)
    auto_make = Column(String(10))
    auto_model = Column(String(10))
    auto_year = Column(Integer)
    fraud_reported = Column(Integer)

