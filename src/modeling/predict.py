import joblib
import typer
import pandas as pd
from loguru import logger
from pathlib import Path
from src.dataset import load_dataframe
import matplotlib.pyplot as plt

from sklearn.metrics import(accuracy_score,
                            confusion_matrix,
                            ConfusionMatrixDisplay,
                            f1_score,
                            precision_score,
                            recall_score,
                            precision_recall_curve,
                            roc_auc_score,
                            RocCurveDisplay)

from src.config import (MODELS_DIR,
                        PROCESSED_DATA_DIR, 
                        DATA_DIR,
                        REPORTS_DIR)


app = typer.Typer()

def components_loader(model_path:Path,
                      ordinal_encoder_path:Path,
                      std_scaler_path:Path) -> tuple[object,
                                                   object,
                                                   object]:
    model = joblib.load(model_path)
    ordinal_encoder = joblib.load(ordinal_encoder_path)
    std_scaler = joblib.load(std_scaler_path)
    
    return model, ordinal_encoder, std_scaler
                         
    
def predictions(test_dataframe: pd.DataFrame, model: object) -> pd.Series:
    logger.info("Starting predictions.")
    
    pred = model.predict(test_dataframe)
    predictions_series = pd.Series(pred, index=test_dataframe.index)
    logger.info("Predictions completed.")
    return predictions_series


def save_pred_df(test_dataframe: pd.DataFrame, prediction: pd.Series, csv_file_path: Path) -> None:
    logger.info("Starting to save predictions to CSV.")
    
    predictions_path: Path = DATA_DIR / "predictions"
    predictions_path.mkdir(parents=True,exist_ok=True)
    dataframe_with_pred = test_dataframe.copy()
    dataframe_with_pred["predictions"] = prediction.reset_index(drop=True)
    
    dataframe_with_pred.to_csv(csv_file_path,index=False)
    
    logger.info(f"""Predictions saved successfully to {csv_file_path} 
                with shape {dataframe_with_pred.shape}""")