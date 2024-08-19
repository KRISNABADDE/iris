import joblib
import typer
import pandas as pd
from pathlib import Path
from loguru import logger
from src.dataset import load_dataframe
from sklearn.linear_model import LogisticRegression
from src.config import MODELS_DIR, PROCESSED_DATA_DIR


app = typer.Typer()

def train(train_df: pd.DataFrame, target_column: str, output_model_file: Path):

    logger.info(f"Starting training of Logistic Regression model with target column '{target_column}'.")

    features = train_df.drop(columns=[target_column],axis=1)
    labels = train_df[target_column]
    
    model = LogisticRegression()
    logger.info("Fitting the model to the training data.")
    model.fit(features, labels)
    logger.success("Modeling training complete.")
    
    joblib.dump(model,output_model_file)
    
    
@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "scaled-data" / "scaled_train_data.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    target_column = 'Species'
):
    
    df = load_dataframe(features_path)
    train(df,target_column,model_path)
   


if __name__ == "__main__":
    app()