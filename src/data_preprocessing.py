import os
import logging
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import string
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from typing import Tuple

log_dir="logs"
logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_preprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_data(train_path:str,test_path:str)->Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df_train=pd.read_csv(train_path)
        df_test=pd.read_csv(test_path)
        logger.debug("raw train and test file loaded")
        return df_train,df_test
    
    except Exception as e:
        logger.error("data loading failed : %s",e)
        raise


def save_data(path:str,train_data:pd.DataFrame,test_data:pd.DataFrame)->None:
    try:
        train_data.to_csv(os.path.join(path,"train_preprocessed.csv"),index=False)
        test_data.to_csv(os.path.join(path,"test_preprocessed.csv"),index=False)
        logger.debug("preprocessing done")

    except Exception as e:
        logger.error("error in saving data: %s",e)
        raise


def preprocess(train_df,test_df,text_column:str,target_column:str)->Tuple[pd.DataFrame, pd.DataFrame]:
    """ remove duplicates,class encoding and transforming text column """
    try:
        logger.debug("preprocessing started")

        train_df=train_df.drop_duplicates(keep="first")
        test_df=test_df.drop_duplicates(keep="first")
        logger.debug("Duplicates droped")

        train_df=train_df.dropna()
        test_df=test_df.dropna()
        logger.debug("nan rows removed")

        encoder=LabelEncoder()
        train_df[target_column]=encoder.fit_transform(train_df[target_column])
        test_df[target_column]=encoder.transform(test_df[target_column])
        logger.debug("targeted column encoding done")

        train_df[text_column]=train_df[text_column].apply(transform_text)
        test_df[text_column]=test_df[text_column].apply(transform_text)
        logger.debug("text transformation done")

        return train_df,test_df
    except Exception as e:
        logger.error("Error during preprocessing %s",e)
        raise
def transform_text(text:str):

    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def main():
    try:
            
        test_data_path="./data/raw/test.csv"
        train_data_path="./data/raw/train.csv"

        train_df,test_df=load_data(train_data_path,test_data_path)
        logger.debug("data loaded properly")

        train_preprocessed_data,test_preprocessed_data=preprocess(train_df,test_df,"text","target")
        logger.debug("Precprocessing Done")

        save_dir="./data"
        path=os.path.join(save_dir,"precessed")
        os.makedirs(path,exist_ok=True)

        save_data(path,train_preprocessed_data,test_preprocessed_data)
        logger.debug("Preprocessed file saved at %s",path)

    except Exception as e:
        logger.error("Preprocessing Failed:%s",e)

if __name__ == "__main__":
    main()