from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BMW Car Price Predictor API",
    description="Predict used BMW car prices based on key features",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],   
)


# Load model once at startup
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Model loading failed: {e}")

# Define the exact input structure (matches your training data)
class CarInput(BaseModel):
    model: Literal["3 Series", "5 Series", "7 Series", "X1", "X3", "X5"] = Field(..., description="BMW model name")
    year: int = Field(..., ge=2000, le=2025, description="Manufacturing year")
    km_driven: int = Field(..., ge=0, description="Kilometers driven")
    fuel_type: Literal["Petrol", "Diesel"] = Field(..., description="Fuel type")
    transmission: Literal["Automatic", "Manual"] = Field(..., description="Transmission type")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "X5",
                "year": 2018,
                "km_driven": 65000,
                "fuel_type": "Diesel",
                "transmission": "Automatic"
            }
        }

@app.get("/")
def home():
    return {"message": "BMW Price Predictor API is live! Visit /docs to test."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_price(car: CarInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # 1. Create base row
    input_dict = car.dict()

    # 2. Manually create one-hot columns (exact names from training)
    dummies = {
        'model_5 Series': 1 if input_dict['model'] == '5 Series' else 0,
        'model_7 Series': 1 if input_dict['model'] == '7 Series' else 0,
        'model_X1':       1 if input_dict['model'] == 'X1' else 0,
        'model_X3':       1 if input_dict['model'] == 'X3' else 0,
        'model_X5':       1 if input_dict['model'] == 'X5' else 0,

        'fuel_type_Petrol': 1 if input_dict['fuel_type'] == 'Petrol' else 0,

        'transmission_Manual': 1 if input_dict['transmission'] == 'Manual' else 0,
    }

    # 3. Build final input row
    row = {
        'year': input_dict['year'],
        'km_driven': input_dict['km_driven'],
        **dummies
    }

    # 4. Create DataFrame with EXACT column order
    input_df = pd.DataFrame([row])[ ['year', 'km_driven', 'model_5 Series', 'model_7 Series', 'model_X1', 'model_X3', 'model_X5', 'fuel_type_Petrol', 'transmission_Manual'] ]

    # Debug prints (keep for now)
    print("Columns sent to model:", input_df.columns.tolist())
    print("Values sent to model:", input_df.iloc[0].values.tolist())

    # 5. Predict
    prediction = model.predict(input_df)[0]

    return {
        "predicted_price": int(round(prediction)),
        "currency": "INR",
        "input_used": input_dict
    }
