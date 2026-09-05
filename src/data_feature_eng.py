from sklearn.feature_extraction.text import  TfidfVectorizer
import os
import logging
import pandas as pd
import yaml

def load_param(param_path:str)->dict:
    try:
        with open(param_path,"r") as file:
            params=yaml.safe_load(file)
            logger.debug("Parametered fetched from %s",param_path)
            return params
        
    except Exception as e:
        logger.error("Paramete fetching failed")
        raise

log_dir="logs"
logger=logging.getLogger("data_feature_eng")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_feature_eng.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def tf_idf(max_feature:int,train_data:pd.DataFrame,test_data:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        tfid = TfidfVectorizer(max_features = max_feature)

        x_train=train_data["text"]
        y_train=train_data["target"]
        x_test=test_data["text"]
        y_test=test_data["target"]

        X_train_bow=tfid.fit_transform(x_train)
        X_test_bow=tfid.transform(x_test)

        train_df=pd.DataFrame(X_train_bow.toarray())
        train_df["label"]=y_train

        test_df=pd.DataFrame(X_test_bow.toarray())
        test_df["label"]=y_test

        return train_df,test_df

    except Exception as e:
        logger.error("tf_idf failed %s",e)
        raise

def main():
    try:
        params=load_param("params.yaml")

        tf_features=params["feature_engineering"]["max_features"]
        
        logger.debug("feature engineering started")

        train_data_path="./data/precessed/train_preprocessed.csv"
        test_data_path="./data/precessed/test_preprocessed.csv"
        train_data=pd.read_csv(train_data_path)
        test_data=pd.read_csv(test_data_path)
        logger.debug("preprocessed train and test data loaded")

        train_data.fillna(" ",inplace=True)
        test_data.fillna(" ",inplace=True)
        logger.debug("nan values filled ")

        train_data,test_data=tf_idf(tf_features,train_data,test_data)
        logger.debug("tf_idf Done")

        path="./data/feature_engineering"
        os.makedirs(path,exist_ok=True)
        train_data.to_csv(os.path.join(path,"train_FE.csv"),index=False)
        test_data.to_csv(os.path.join(path,"test_FE.csv"),index=False)
        logger.debug("featured train and test data saved to %s",path)

        
    except Exception as e:
        logger.error("feature engineering failed : %s",e)


if __name__=="__main__":
    main()
