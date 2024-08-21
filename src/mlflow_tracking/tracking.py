import mlflow

def mlflow_tracking(params:dict,metrics:dict):
    mlflow.set_tracking_uri("http://127.0.0.1:8000")

    if mlflow.is_tracking_uri_set():
        mlflow.set_experiment(experiment_name="iris-logistic")
        with mlflow.start_run(run_name="first-run") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics=metrics)
    else:
        raise("Please set tracing url")
        