import typer
import zipfile
import pandas as pd
from loguru import logger
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def extract_files(zipfile_path: Path)-> None:
    
    extract_dir = RAW_DATA_DIR / "extracted"
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
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
    
    
@app.command()
def main(
    zip_file_path:Path = RAW_DATA_DIR / "iris-dataset.zip",
    input_path: Path =  RAW_DATA_DIR / 'extracted' / 'iris.csv',
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    
        extract_files(zip_file_path)
        df = load_dataframe(input_path)
    
        df = drop_column(df,['Id'])
    
        save_dataframe(df,output_path)
        logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
