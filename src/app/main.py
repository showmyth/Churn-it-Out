from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.serving.inference import predict  # Core ML inference logic


app = FastAPI(
    title="Churn it Out",
    description="An API pointing to a serving model used for churn prediction (trained on Telco Churn Dataset)",
    version = "1.0.0"
)


# === ROOT ===
@app.get("/")
def root():
    """ Health Check """
    return {"status": "ok"}

# === DATA SCHEMA ===
class CustomerData(BaseModel):
    # Self Disclosure
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" 
    Dependents: str            # "Yes" or "No" 
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No" or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic" or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check" ...
    
    # Numeric features
    tenure: int                # No. of months with company
    MonthlyCharges: float      # Monthly charges (in $)
    TotalCharges: float        # Total charges (to date)

# === MAIN PREDICTION API === 
@app.post("/predict")
def get_prediction(data: CustomerData) -> dict:
    """
    The Prediction Endpoint for churn predictions.
    
    Functions :
    - Receives customer data (wherein data is check through Pydantic)
    - Calls inference.py -> predict() to predict whether given input(s) will churn or not
    - Returns churn prediction as JSON object

    Expected Response:
    - {"prediction": "Yes"} or {"prediction": "No"}
    - {"error": "error_message"} if prediction fails
    """

    try:
        result = predict(data.model_dump())
        return {"prediction" : result}
    
    except Exception as exc:
        return {"error" : str(exc)}

    


