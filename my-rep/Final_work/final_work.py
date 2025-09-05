from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import dill

app = FastAPI(title="Cars Predictor")


MODEL_PATH = Path(__file__).with_name("pipe.pkl")

model = None
metadata = {}
try:
    with open(MODEL_PATH, "rb") as f:
        obj = dill.load(f)
        if isinstance(obj, dict) and "model" in obj:
            model = obj["model"]
            metadata = obj.get("metadata", {})
        else:
            model = obj
except FileNotFoundError:
    
    model = None


class CarItem(BaseModel):
    id: int | None = None
    price: float | int | None = None
    year: int | None = None
    manufacturer: str | None = None
    model: str | None = None
    fuel: str | None = None
    odometer: float | int | None = None
    title_status: str | None = None
    transmission: str | None = None
    state: str | None = None
    description: str | None = None
    image_url: str | None = None
    lat: float | None = None
    long: float | None = None
    posting_date: str | None = None
    region: str | None = None
    region_url: str | None = None
    url: str | None = None

@app.get("/health")
def health():
    return {"ok": model is not None, "has_metadata": bool(metadata)}

@app.post("/predict")
def predict(item: CarItem):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded at {MODEL_PATH.name}")
    df = pd.DataFrame([item.model_dump()])
    try:
        y = model.predict(df)
    except Exception as e:
        
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
   
    val = y[0] if hasattr(y, "__len__") else y
    try:
        val = int(val)
    except Exception:
        pass
    return {"prediction": val}

