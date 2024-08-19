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
                         
    
   
