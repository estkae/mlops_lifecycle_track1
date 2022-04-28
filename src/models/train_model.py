# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:25 2022

@author: LENOVO PC
"""

#import json
from __future__ import with_statement
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import json


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["random_forest"]["max_depth"]
    n_estimators=config["random_forest"]["n_estimators"]
    paramsf=config["reports"]["params"]
    scoresf=config["reports"]["scores"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    with open(paramsf,"w") as f:
        params={"max_depth":max_depth,"n_estimators":n_estimators}
        json.dump(params,f,indent = 4)


################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    client = MlflowClient()

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        print(mlops_run.info.run_id)
        model = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')

        with open(scoresf,"w") as f:
            scores={"accuracy":accuracy,"precision":precision,"recall":recall,"f1score":f1score}
            json.dump(scores,f,indent = 4)

        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)
       
        signature = infer_signature(train_x, model.predict(train_x))

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        #tracking_url_type_store = "file1"

        if tracking_url_type_store != "file":
            #client.create_registered_model(mlflow_config["registered_model_name"])
            '''result = client.create_model_version(
                name=mlflow_config["registered_model_name"],
                source="mlruns/0/"+mlops_run.info.run_id+"/artifacts/model",
                run_id=mlops_run.info.run_id)
            model_uri="runs:/"+mlops_run.info.run_id+"artifacts/model"
            result=mlflow.register_model(model_uri,mlflow_config["registered_model_name"])
            print(result.version)'''

            """             mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"]) """
            
            mlflow.sklearn.log_model(
                model, 
                "model", 
                signature=signature)

        else:
            #mlflow.sklearn.log_model(model, "model",signature=signature)
            mlflow.sklearn.log_model(sk_model=model,artifact_path="model",registered_model_name=mlflow_config["registered_model_name"])

 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)