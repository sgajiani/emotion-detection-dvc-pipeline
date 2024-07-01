import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml
import logging
from typing import Tuple

# configure logging
logger = logging.getLogger('model_building')
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

# read params
def load_params(param_path: str) -> Tuple[int, float]:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)['train_model']

        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']

        return n_estimators, learning_rate

    except FileNotFoundError:
        logger.error(f"Error: The file {param_path} was not found.")
        raise
    except KeyError as e:
        logger.error(f"Error: Key {e} not found in the parameter file.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error: YAML error occurred: {e}")
        raise



def read_train_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df

    except FileNotFoundError:
        logger.error(f"Error: The file {data_path} was not found.")
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


def build_model(df_train: pd.DataFrame, n_estimators: int, learning_rate: float, model_filename: str) -> None:
    try:
        X_train = df_train.iloc[:,0:-1].values
        y_train = df_train.iloc[:,-1].values

        # fit the model
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train, y_train)

        # save the model
        pickle.dump(model, open(model_filename, 'wb'))

    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
    except IndexError as ie:
        logger.error(f"IndexError occurred: {ie}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def main():
    n_estimators, learning_rate = load_params('params.yaml')

    vectorized_data_path = os.path.join("data", "features")
    train_datapath = os.path.join(vectorized_data_path, "train_tfidf.csv")
    
    df_train = read_train_data(train_datapath)

    build_model(df_train, n_estimators, learning_rate, model_filename='models\gbc.pkl')



if __name__ == '__main__':
    main()