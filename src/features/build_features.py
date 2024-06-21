import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging
from typing import Tuple

# configure logging
logger = logging.getLogger('build_features')
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

def load_params(param_path: str) -> int:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['build_features']['max_features']
        return max_features
    
    except FileNotFoundError:
        logger.error(f"Error: The file {param_path} was not found.")
        raise
    except KeyError as e:
        logger.error(f"Error: Key {e} not found in the parameter file.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error: YAML error occurred: {e}")
        raise

    

def read_data(train_datapath: str, test_datapath: str)-> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_processed_data = pd.read_csv(train_datapath)
        test_processed_data = pd.read_csv(test_datapath)

        train_processed_data.fillna('', inplace=True)
        test_processed_data.fillna('', inplace=True)

        return train_processed_data, test_processed_data


    except FileNotFoundError:
        logger.error(f"Error: The file was not found.")
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


def split_data(train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        X_train = train_processed_data['content'].values
        y_train = train_processed_data['sentiment'].values

        X_test = test_processed_data['content'].values
        y_test = test_processed_data['sentiment'].values

        return X_train, X_test, y_train, y_test
    
    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Invalid attribute access: {e}")
        raise
    except NameError as e:
        logger.error(f"Data not defined: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def vectorize_data(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        # Apply Bag of Words 
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit the vectorizer to training data nd transform
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform the test data
        X_test_bow = vectorizer.transform(X_test)

        return X_train_bow, X_test_bow
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    


def save_vectorized_data(df_train_bow: pd.DataFrame, df_test_bow: pd.DataFrame, data_path: str) -> None:
    try:
        
        os.makedirs(data_path, exist_ok=True)

        df_train_bow.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        df_test_bow.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

    except PermissionError:
        logger.error(f"Error: Permission denied while creating directory {data_path}.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred while saving data: {e}")
        raise



def main():
    max_features = load_params('params.yaml')

    processed_data_path = os.path.join("data", "processed")
    train_datapath = os.path.join(processed_data_path, "train_processed.csv")
    test_datapath = os.path.join(processed_data_path, "test_processed.csv")

    train_processed_data, test_processed_data = read_data(train_datapath, test_datapath)

    X_train, X_test, y_train, y_test = split_data(train_processed_data, test_processed_data)

    X_train_bow, X_test_bow = vectorize_data(X_train, X_test, max_features)

    # Combine X and y to create a df
    df_train_bow = pd.DataFrame(X_train_bow.toarray())
    df_train_bow['label'] = y_train

    df_test_bow = pd.DataFrame(X_test_bow.toarray())
    df_test_bow['label'] = y_test

    vectorized_data_path = os.path.join("data", "features")
    save_vectorized_data(df_train_bow, df_test_bow, data_path=vectorized_data_path)


if __name__ == '__main__':
    main()
