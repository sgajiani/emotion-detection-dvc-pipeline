import numpy as np
import pandas as pd
import os
from typing import Tuple
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

nltk.download('wordnet')
nltk.download('stopwords')

# ofigure logging
logger = logging.getLogger('preprocess_data')
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


def read_data(train_datapath: str, test_datapath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_datapath)
        test_data = pd.read_csv(test_datapath)

        return train_data, test_data
    
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

# transform the data
def lemmatize_word(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        lematized_text = [lemmatizer.lemmatize(word) for word in text]
        return " " .join(lematized_text)
    
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        raise


def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        no_sw_text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(no_sw_text)
    
    except Exception as e:
        logger.error(f"Error in removing stop words: {e}")
        raise



def remove_numbers(text: str) -> str:
    try:
        no_num_text= ''.join([word for word in text if not word.isdigit()])
        return no_num_text
    
    except Exception as e:
        logger.error(f"Error in removing numbers: {e}")
        raise


def convert_to_lowercase(text: str) -> str:
    try:
        text = text.split()
        lower_text = [word.lower() for word in text]
        return " " .join(lower_text)
    
    except Exception as e:
        logger.error(f"Error in converting to lowercase: {e}")
        raise



def remove_punctuation(text: str) -> str:
    try:
        no_punc_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        no_punc_text = no_punc_text.replace('؛',"", )
        ## remove extra whitespace
        no_punc_text = re.sub('\s+', ' ', no_punc_text)
        no_punc_text =  " ".join(no_punc_text.split())
        return no_punc_text.strip()
    
    except Exception as e:
        logger.error(f"Error in removing punctuation: {e}")
        raise




def remove_url(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    except Exception as e:
        logger.error(f"Error in removing URLs: {e}")
        raise


def remove_small_sentence(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        return df
    
    except Exception as e:
        logger.error(f"Error in removing small sentences: {e}")
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lambda content : convert_to_lowercase(content))
        df.content = df.content.apply(lambda content : remove_stop_words(content))
        df.content = df.content.apply(lambda content : remove_numbers(content))
        df.content = df.content.apply(lambda content : remove_punctuation(content))
        df.content = df.content.apply(lambda content : remove_url(content))
        df.content = df.content.apply(lambda content : lemmatize_word(content))
        return df
    
    except Exception as e:
        logger.error(f"Error in normalizing text: {e}")
        raise


def normalize_sentence(sentence: str) -> str:
    try:
        sentence = convert_to_lowercase(sentence)
        sentence = remove_stop_words(sentence)
        sentence = remove_numbers(sentence)
        sentence = remove_punctuation(sentence)
        sentence = remove_url(sentence)
        sentence = lemmatize_word(sentence)
        return sentence
    
    except Exception as e:
        logger.error(f"Error in saving processed data: {e}")
        raise


def save_processed_data(train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame, data_path: str) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

    except Exception as e:
        logger.error(f"Error in saving processed data: {e}")
        raise


def main():
    raw_data_path = os.path.join("data", "raw")
    train_datapath = os.path.join(raw_data_path, "train.csv")
    test_datapath = os.path.join(raw_data_path, "train.csv")
    
    train_data, test_data = read_data(train_datapath, test_datapath)

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    processed_data_path = os.path.join("data", "processed") 
    save_processed_data(train_processed_data, test_processed_data, data_path=processed_data_path)



if __name__ == "__main__":
    main()

