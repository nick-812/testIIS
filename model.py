import mlflow

print("jdjdkdfs")

mlflow.set_tracking_uri("https://dagshub.com/nick-812/testIIS.mlflow")

os.environ['MLFLOW_TRACKING_USERNAME'] = 'nick-812'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f37e3703f5ec08bb5db0beceeeb610ef344dc6da'

mlflow.sklearn.autolog()