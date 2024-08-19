import typer
import zipfile
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def extract_files(RAW_DATA_DIR: Path, file_name: str)-> None:
    
    zip_file_path = RAW_DATA_DIR / file_name
    
    extract_dir = RAW_DATA_DIR / "extracted"
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    logger.success("Unzip datafile complete.")
    

def load_dataframe(csv_file_path:Path) -> pd.DataFrame:

    df = pd.read_csv(csv_file_path)
    logger.success(f"Dataframe loaded. with shape of {df.shape}")
    return df


def drop_column(dataframe: pd.DataFrame, column_name: list) -> pd.DataFrame:
    df = dataframe.drop(columns=column_name)
    logger.info(f"Columns dropped: {', '.join(column_name)}")
    return df


def save_dataframe(dataframe: pd.DataFrame, output_dir_path: Path) -> None:
   
    dataframe.to_csv(output_dir_path, index=False)
    logger.info(f"DataFrame saved successfully at {output_dir_path} with shape {dataframe.shape}")
    