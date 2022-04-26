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

    run = MlflowClient().search_runs(
        experiment_ids="1",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.accuracy DESC"]
        )[0]
    ###
    print(run)
    print("Iam here now")
    print(type(run))
    print(run.info.run_id)
    print("Iam here too")

    client = MlflowClient()
    client.create_registered_model(model_name)
    stage = 'Staging'
    
    print("model got registered")
  

    client = MlflowClient()
    result = client.create_model_version(
                name=model_name,
                source="mlruns/1/"+run.info.run_id+"/artifacts/model",
                run_id=run.info.run_id)

    print(result.version)

    client = MlflowClient()
    client.update_model_version(
    name=model_name,
    version=1,
    description="This model version is a scikit-learn random forest containing 150 decision trees")

    #model_uri="runs:/"+run.info.run_id+"artifacts/model"
    #result=mlflow.register_model(model_uri,mlflow_config["registered_model_name"])

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )

    print("Here are our registered models")

    from pprint import pprint

    client = MlflowClient()
    for rm in client.list_registered_models():
        pprint(dict(rm), indent=4)

    model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
    )

    '''client.transition_model_version_stage(
                name=model_name,
                stage="Staging")'''

    '''client = MlflowClient(tracking_uri=remote_server_uri)
    filter_string = "name='{}'".format(model_name)
    print(filter_string)
    print(mlflow.client.search_model_versions(f"name='{model_name}'"))
    results = client.search_model_versions(filter_string)
    print(results)
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        print(mv)
        print('iam here')
        if mv["run_id"] == max_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )'''       
   
    loaded_model = mlflow.pyfunc.load_model('mlruns/1/'+run.info.run_id+'/artifacts/model')
    #loaded_model = mlflow.pyfunc.load_model(logged_model)

    joblib.dump(loaded_model, model_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)