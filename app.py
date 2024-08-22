from fastapi import FastAPI
import pandas as pd
from src.modeling.predict import components_loader
from data_validation import test
from src.config import COMPONENTS_DIR, MODELS_DIR
model_path = MODELS_DIR / "model.joblib"
ord_encoder_path = COMPONENTS_DIR / "ord_encoder.joblib"
stdscaler_path = COMPONENTS_DIR / "stdscaler.joblib"



app = FastAPI()

@app.get('/')
def helloword():
    return "hello fast"

@app.post("/predict/")
async def predict(test: test):
    # Convert input data to DataFrame
    model,ord_encoder, stdscaler = components_loader(model_path,ord_encoder_path,
                                                stdscaler_path)
    data = {
        'SepalLengthCm': [test.SepalLengthCm],
        'SepalWidthCm': [test.SepalWidthCm],
        'PetalLengthCm': [test.PetalLengthCm],
        'PetalWidthCm': [test.PetalWidthCm]
    }
    
    # Transform the data (using the scaler and encoder as necessary)
    df = pd.DataFrame(data)
    columns = df.columns.tolist()
    df = stdscaler.transform(df)
    features = pd.DataFrame(df, columns=columns)
    
    # Make prediction
    prediction = model.predict(features)
    
    prediction = ord_encoder.inverse_transform(prediction.reshape(-1, 1))
    
    return {"prediction": prediction.flatten().tolist()[0]}