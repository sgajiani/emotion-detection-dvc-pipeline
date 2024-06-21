import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple
import logging

# cofigure logging
logger = logging.getLogger('make_dataset')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(param_path: str) -> float:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['make_dataset']['test_size']
        return test_size
    
    except FileNotFoundError:
        logger.error(f"Error: The file {param_path} was not found.")
        raise
    except KeyError as e:
        logger.error(f"Error: Key {e} not found in the parameter file.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error: YAML error occurred: {e}")
        raise



def read_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df
    
    except FileNotFoundError:
        logger.error(f"Error: The file {filepath} was not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Error: No data found in the file.")
        raise
    except pd.errors.ParserError:
        logger.error("Error: Error parsing the file.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred while processing data: {e}")
        raise



def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    try:
        df_n = df.copy()
        df_n.drop('tweet_id', axis=1, inplace=True)
        df_n = df_n[df_n['sentiment'].isin(['happiness', 'sadness'])]
        codes, uniques = pd.factorize(df_n['sentiment'])
        df_n['sentiment'] = pd.Categorical(codes)
        return df_n, uniques
    
    except KeyError as e:
        logger.error(f"Error: Key {e} not found in the parameter file.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred while processing data: {e}")
        raise


def save_splitted_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

    except PermissionError:
        logger.error(f"Error: Permission denied while creating directory {data_path}.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred while saving data: {e}")
        raise


def main():
    # read params
    test_size = load_params('params.yaml')
    
    # read csv
    df = read_data('./data/external/tweet_emotions.csv')
    
    # process data
    df_processed, uniques = process_data(df)
    
    # split data
    train_data, test_data = train_test_split(df_processed, test_size=test_size, random_state=42)

    # save splitted data
    data_path = os.path.join("data", "raw")
    save_splitted_data(train_data, test_data, data_path)


if __name__ == "__main__":
    main()
