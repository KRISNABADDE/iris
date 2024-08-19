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
    
    ord_encoder_file_path = components_dir / "ord_encoder.joblib"
    joblib.dump(ord_encoder,ord_encoder_file_path)
    
    logger.info("Conversion complete. Returning the transformed DataFrame.")
    
    return dataframe


def data_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    try:
        parameters = safe_load(open('params.yaml','r'))['data_ingestion']
        logger.log_message('Parameters read successfully')
    except FileNotFoundError as e:
        logger.log_message('Parameters file missing')
    
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    
    logger.log_message(f'Parameters : test_size={test_size}  random_state={random_state}')
    
    train_data, test_data = train_test_split(dataframe,
                                             test_size=test_size,
                                             random_state=random_state)
    
    logger.info(f"Data split into train data with shape {train_data.shape} and test data with shape {test_data.shape}")
    
    return train_data, test_data


def scale_data(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame,
               columns: list, target_column: str, output_dir: Path,
               components_dir: Path) -> None:
    
    if not columns:
        raise ValueError("The columns list is empty or None.")

    for column in columns:
        if column not in train_dataframe.columns:
            raise ValueError(f"Column '{column}' does not exist in the training DataFrame.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_dataframe[columns])
    test_scaled = scaler.transform(test_dataframe[columns])
    
    scaler_file_path = components_dir / "stdscaler.joblib"
    joblib.dump(scaler,scaler_file_path)
    
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns)
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns)
    
    train_scaled_df[target_column] = train_dataframe[target_column].reset_index(drop=True)
    test_scaled_df[target_column] = test_dataframe[target_column].reset_index(drop=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output_file = output_dir / 'scaled_train_data.csv'
    test_output_file = output_dir / 'scaled_test_data.csv'

    train_scaled_df.to_csv(train_output_file, index=False)
    test_scaled_df.to_csv(test_output_file, index=False)

    logger.info(f"Scaled training data saved to '{train_output_file}' with shape {train_scaled_df.shape}")
    logger.info(f"Scaled test data saved to '{test_output_file}' with shape {test_scaled_df.shape}")
    


@app.command()
def main(
    target_column = 'Species',
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "scaled-data"
    
):
    df = load_dataframe(input_path)
    
    dataframe_encoded = categorical_to_numerical(df,target_column)
    
    train, test = data_split(dataframe_encoded)
    
    columns = train.columns.tolist()
    columns.remove(target_column)
    
    scale_data(train,test,columns,target_column,output_path)
    
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()