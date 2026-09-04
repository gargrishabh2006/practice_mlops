import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

log_path=os.path.join(log_dir,"data_ingestion.log")
file_handler=logging.FileHandler(log_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(url:str)->pd.DataFrame:
    """Load data from link"""
    try:
        df=pd.read_csv(url)
        logger.debug("data loaded")
        return df
    except Exception as e:
        logger.error("Failed to load data: %s",e)
        raise
    

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,path:str)->None:
    """ Save the Train and Test dataset """
    try:
       raw_data_path= os.path.join(path,"raw")
       os.makedirs(raw_data_path,exist_ok=True)
       train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
       test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
       logger.debug("Train and Test data saved to %s",raw_data_path)

    except Exception as e:
        logger.error("Data not Saved %s",e)
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """Column Dropping and Naming Done"""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug("Preprocessing Done")
        return df
    except Exception as e:
        logger.error("data preprocessing not done: %s",e)
        raise
    

def main():
    try:
        test_size=.2
        dataset_link="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df=load_data(dataset_link)

        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,random_state=2,test_size=test_size)

        save_data(train_data,test_data,"./data")

    except Exception as e:
        logger.error("failed to complete data ingestion %s",e)
        print(f"error:{e}")

if __name__=="__main__":
    main()


