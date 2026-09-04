import os
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score
import pickle
import json
import logging

log_dir="logs"
logger=logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"model_evaluation.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_model(model_path:str):
    try:
        with open(model_path,"rb") as file:
            model=pickle.load(file)
        logger.debug("model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load the model")
        raise
    

def evaluation(model,test_data:pd.DataFrame)->dict:
    try:
        y_pred=model.predict(test_data.iloc[:,:-1])
        accuracy=accuracy_score(test_data.iloc[:,-1],y_pred)
        precision=precision_score(test_data.iloc[:,-1],y_pred)
        recall =recall_score(test_data.iloc[:,-1],y_pred)
        metrics_dict = {
                   'accuracy': accuracy,
                   'precision': precision,
                   'recall': recall
        }
        logger.debug("model evaluation metrics calculated")
        return metrics_dict
    
    except Exception as e:
        logger.error("Error during model Evaluation: %s",e)
        raise

def metrics_save(metrics:dict,metrics_path:str):
    try:
        os.makedirs(os.path.dirname(metrics_path),exist_ok=True)
        with open(metrics_path,"w") as file:
            json.dump(metrics,file,indent=4)

    except Exception as e:
        logger.error("Error in saving the metrics")
        raise
    

def main():
    try:

        test_data_path="./data/feature_engineering/test_FE.csv"
        test_data=pd.read_csv(test_data_path)
        logger.debug("test data loaded")

        model_path="./models/models.pkl"
        model=load_model(model_path)
        logger.debug("Model loaded")

        metrics=evaluation(model,test_data)
        logger.debug("metrics evaluated")

        metrics_path="./metrics/metrics.json"
        metrics_save(metrics,metrics_path)
        logger.debug("Metrics saved to %s",metrics_path)

    except Exception as e:
        logger.error("model evaluation failed")

if __name__=="__main__":
    main()