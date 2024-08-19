import typer
import joblib
import pandas as pd
from yaml import safe_load
from loguru import logger
from pathlib import Path
from src.dataset import load_dataframe
from src.config import PROCESSED_DATA_DIR, COMPONENTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler,
                                   OrdinalEncoder)
app = typer.Typer()


def categorical_to_numerical(dataframe: pd.DataFrame, target_column: str,
                             components_dir: Path) -> pd.DataFrame:
    
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the DataFrame.")
    
    logger.info(f"Applying ordinal encoding to the target column '{target_column}'.")
    ord_encoder = OrdinalEncoder()
    dataframe[target_column] = ord_encoder.fit_transform(dataframe[[target_column]])
    
    components_file_path = components_dir / "ord_encoder.joblib"
    joblib.dump(ord_encoder,components_file_path)
    
    logger.info("Conversion complete. Returning the transformed DataFrame.")
    
    return dataframe


def data_split(dataframe: pd.DataFrame, test_size: float,
               random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_data, test_data = train_test_split(dataframe,
                                             test_size=test_size,
                                             random_state=random_state)
    
    logger.info(f"Data split into train data with shape {train_data.shape} and test data with shape {test_data.shape}")
    
    return train_data, test_data



