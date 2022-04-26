#import os
import argparse
import pandas as pd
from loan_data import read_params


def processing (tr_df,data_path):

    #converting categorical values to numbers

    to_numeric = {'Male': 1, 'Female': 2,'Yes': 1, 'No': 2,'Graduate': 1, 'Not Graduate': 2,'Urban': 3, 'Semiurban': 2,'Rural': 1,'Y': 1, 'N': 0,'3+': 3}

    # adding the new numeric values from the to_numeric variable to both datasets
    tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
    #te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)

    # convertind the Dependents column
    Dependents_ = pd.to_numeric(tr_df.Dependents)
    #Dependents__ = pd.to_numeric(te_df.Dependents)

    # dropping the previous Dependents column
    tr_df.drop(['Dependents'], axis = 1, inplace = True)
    #te_df.drop(['Dependents'], axis = 1, inplace = True)

    # concatination of the new Dependents column with both datasets
    tr_df = pd.concat([tr_df, Dependents_], axis = 1)
    #te_df = pd.concat([te_df, Dependents__], axis = 1)
    tr_df.to_csv(data_path)
 



def process_data(config_path):
    """
    impute and processing data and save it in the data/processed folder
    input: config path 
    output: save splitted files in output folder
    """
    config = read_params(config_path)
    test_data_path = config["interim_data_config"]["test_data_csv"] 
    train_data_path = config["interim_data_config"]["train_data_csv"]
    processed_test_data_path = config["processed_data_config"]["test_data_csv"] 
    processed_train_data_path = config["processed_data_config"]["train_data_csv"] 

    interim_df=pd.read_csv(train_data_path)
    processing (interim_df,processed_train_data_path)
    interim_df=pd.read_csv(test_data_path)
    processing (interim_df,processed_test_data_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    process_data(config_path=parsed_args.config)

