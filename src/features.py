import typer
import joblib
import pandas as pd
from loguru import logger
from pathlib import Path
from src.dataset import load_dataframe
from src.config import PROCESSED_DATA_DIR, COMPONENTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler,
                                   OneHotEncoder,
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


