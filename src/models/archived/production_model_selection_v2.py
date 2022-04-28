import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    #all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]
    #runs = mlflow.search_runs(experiment_ids=all_experiments)
    #print(runs)
    #max_accuracy = max(runs["metrics.accuracy"])
    #print(max_accuracy)
    #max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]
    #print(max_accuracy_run_id)
###
    from mlflow.tracking.client import MlflowClient
    from mlflow.entities import ViewType

    client = MlflowClient()

    max_run = client.search_runs(
        experiment_ids="1",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.accuracy DESC"]
        )[0]
    ###
    #print(max_run.info.run_id)
    runids=client.list_run_infos(experiment_id="1",run_view_type=ViewType.ALL)
    for run in runids:
        m=dict(run)["run_id"]
        if m==max_run.info.run_id:
            client.create_registered_model(model_name+"_"+str(m))
            result = client.create_model_version(
                        name=model_name+"_"+str(m),
                        source="mlruns/1/"+str(m)+"/artifacts/model",
                        run_id=m)
            print(result.version)
            client.transition_model_version_stage(
                    name=model_name+"_"+str(m),
                    version=result.version,
                    stage="Production" )
        else:
            client.create_registered_model(model_name+"_"+str(m))
            result = client.create_model_version(
                        name=model_name+"_"+str(m),
                        source="mlruns/1/"+str(m)+"/artifacts/model",
                        run_id=m)
            print(result.version)
            client.transition_model_version_stage(
                    name=model_name+"_"+str(m),
                    version=result.version,
                    stage="Staging" )





 
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)