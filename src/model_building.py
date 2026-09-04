import logging
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

log_dir="logs"
logger=logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def model_training(train_data:pd.DataFrame,params:dict)-> RandomForestClassifier:
    try:
        if(train_data.iloc[:,:-1].shape[0]!=train_data.iloc[:,-1].shape[0]):
            raise ValueError("shaped mismatched !!!!")
        
        rf=RandomForestClassifier(n_estimators=params["n_estimators"],random_state=params["random_state"])

        logger.debug("Model training started")
        rf.fit(train_data.iloc[:,:-1],train_data.iloc[:,-1])
        logger.debug("model training completed")
        return rf
    except Exception as e:
        logger.error("error during model training :%s",e)
        raise

def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(model,file)
        logger.debug("Model saved to %s",file_path)

    except Exception as e:
        logger.debug("error while saving the model : %s",e)
        raise

def main():
    try:
        params={"n_estimators":50,"random_state":2}
        train_data= pd.read_csv("./data/feature_engineering/train_FE.csv")
        logger.debug("train_data loaded")

        rf=model_training(train_data,params)
        model_save_path="models/models.pkl"
        save_model(rf,model_save_path)
        
    except Exception as e:
        logger.error("model building failed %s",e)

if __name__=="__main__":
    main()
