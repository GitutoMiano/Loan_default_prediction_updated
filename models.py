from pydantic import BaseModel
from fastapi import FastAPI, Form

# Define your data model
class LoanRequest(BaseModel):
    loan_limit: str
    gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    credit_worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: int
    rate_of_interest: float
    interest_rate_spread: float
    upfront_charges: float
    term: int
    neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    property_value: float
    construction_type: str
    occupancy_type: str
    secured_by: str
    total_units: str
    income: float
    credit_type: str
    credit_Score: int
    co_applicant_credit_type: str
    age: str
    submission_of_application: str
    ltv: float
    region: str
    security_Type: str
    dtir1: int
