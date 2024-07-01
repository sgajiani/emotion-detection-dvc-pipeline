import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
from typing import Tuple

# configure logging
logger = logging.getLogger('predict_model')
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


def read_test_data(data_path: str) -> pd.DataFrame:
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


def predict(X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        model = pickle.load(open('models/gbc.pkl', 'rb'))

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]

        return y_pred, y_pred_proba
    
    except FileNotFoundError:
        logger.error(f"Error: The model file was not found.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray):
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        return accuracy, precision, recall, auc

    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
    except IndexError as ie:
        logger.error(f"IndexError occurred: {ie}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")



def main():
    vectorized_data_path = os.path.join("data", "features")
    test_datapath = os.path.join(vectorized_data_path, "test_tfidf.csv")

    df_test = read_test_data(test_datapath)

    X_test = df_test.iloc[:,0:-1].values
    y_test = df_test.iloc[:,-1].values   

    y_pred, y_pred_proba = predict(X_test)

    accuracy, precision, recall, auc = evaluate_model(y_test, y_pred, y_pred_proba)

    # create metrics json file
    metrics_dict = {
        'accuracy':accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    with open('reports/metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)


if __name__ == '__main__'        :
    main()