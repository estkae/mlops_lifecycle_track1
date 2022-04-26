# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:35:35 2022

@author: LENOVO PC
"""

#import os
import argparse
import pandas as pd
from loan_data import read_params
from sklearn.model_selection import train_test_split

def impute_data(tr_df,cols):
    #filling the missing data
    #null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
    null_cols = cols


    for col in null_cols:
        #print(f"{col}:\n{tr_df[col].value_counts()}\n","-"*50)
        tr_df[col] = tr_df[col].fillna(tr_df[col].dropna().mode().values[0] )   

        
    tr_df.isnull().sum().sort_values(ascending=False)
    #print("After filling missing values\n\n","#"*50,"\n")
    #for col in null_cols:
      #  print(f"\n{col}:\n{tr_df[col].value_counts()}\n","-"*50)
    return(tr_df)


def split_data(df,train_data_path,test_data_path,split_ratio,random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")    

def split_and_saved_data(config_path):
    """
    split the train dataset(data/raw) and save it in the data/processed folder
    input: config path 
    output: save splitted files in output folder
    """
    config = read_params(config_path)
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    test_data_path = config["interim_data_config"]["test_data_csv"] 
    train_data_path = config["interim_data_config"]["train_data_csv"]
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]
    model_var = config["raw_data_config"]["model_var"]
    target=config["raw_data_config"]["target"]
    model_var.remove(target)

    raw_df=pd.read_csv(raw_data_path)
    
    raw_df = impute_data(raw_df,model_var)
    split_data(raw_df,train_data_path,test_data_path,split_ratio,random_state)
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)