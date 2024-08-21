import joblib
import typer
import pandas as pd
from loguru import logger
from pathlib import Path
from src.dataset import load_dataframe
from src.mlflow_tracking.tracking import mlflow_tracking
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
                         
    
def predictions(test_dataframe: pd.DataFrame, model: object,
                std_scaler: object) -> pd.Series:
    logger.info("Starting predictions.")
    columns = test_dataframe.columns.tolist()
    scaled_features = std_scaler.transform(test_dataframe[columns])
    features = pd.DataFrame(scaled_features, columns=columns)
    pred = model.predict(features)
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
    

def val_metrics(prediction: pd.Series, labels: pd.Series,
                metrix_file_dir: Path, metrics_plot_dir: Path) -> pd.DataFrame:
    
    logger.info("Starting the calculation of evaluation metrics.")
    
    accuracy = accuracy_score(labels, prediction)
    confusion_mat = confusion_matrix(labels, prediction)
    precision_s = precision_score(labels, prediction, average='macro')
    recall_s = recall_score(labels, prediction, average='macro')
    f1_s = f1_score(labels, prediction, average='macro')
    
    try:
        roc_auc_s = roc_auc_score(labels, prediction, multi_class="ovr", average="macro")
        logger.info("ROC AUC Score calculated successfully.")
    except ValueError:
        roc_auc_s = None
        logger.warning("ROC AUC Score could not be calculated.")
    
    metrix_file_dir.mkdir(parents=True, exist_ok=True)
    metrics_plot_dir.mkdir(parents=True, exist_ok=True)
    
    confusion_plot_path = metrics_plot_dir / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(confusion_plot_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved at {confusion_plot_path}.")
    
    if roc_auc_s is not None:
        roc_plot_path = metrics_plot_dir / "roc_curve.png"
        RocCurveDisplay.from_predictions(labels, prediction)
        plt.title("ROC Curve")
        plt.savefig(roc_plot_path)
        plt.close()
        logger.info(f"ROC curve plot saved at {roc_plot_path}.")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision_s, recall_s, f1_s]
    }
    
    if roc_auc_s is not None:
        metrics_data['Metric'].append('ROC AUC')
        metrics_data['Value'].append(roc_auc_s)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Metric', inplace=True)
    
    metrics_csv_path = metrix_file_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path)
    logger.info(f"Metrics DataFrame saved at {metrics_csv_path}.")
    
    logger.info("Evaluation metrics calculation and saving completed.")
    
    return metrics_df


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "scaled-data" / "test_data.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    ordinal_encoder_path: Path = MODELS_DIR / "components" / "ord_encoder.joblib",
    std_scaler_path: Path = MODELS_DIR / "components" / "stdscaler.joblib",
    predictions_path: Path = DATA_DIR / "predictions" / "test_predictions.csv",
    target_column = "Species",
    metrix_path: Path = REPORTS_DIR / "metrix",
    metrix_plots_path: Path = REPORTS_DIR / "metrix" / "graphs"
    
):
    model, ordinal_encoder, std_scale = components_loader(model_path,
                                                          ordinal_encoder_path,
                                                          std_scaler_path)
    logger.info("model loaded")
    logger.info("ordinal_encoder loaded")
    logger.info("std_scale loaded")
    
    df = load_dataframe(features_path)
    pred = predictions(df.drop(columns=[target_column]),model,std_scale)
    save_pred_df(df,pred,predictions_path)
    metrics_df = val_metrics(pred,df[target_column],metrix_path,metrix_plots_path)
    metrics = metrics_df.to_dict()['Value']
    mlflow_tracking(model.get_params(),metrics=metrics)

if __name__ == "__main__":
    app()