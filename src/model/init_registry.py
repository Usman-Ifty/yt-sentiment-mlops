import mlflow
from mlflow.tracking import MlflowClient
import os

# Configuration from your notebook output
RUN_ID = "bc0f3e0185704b4586b46d1006b6a51a"
MODEL_NAME = "yt-sentiment-bert"

# Physical path to the model artifacts found in your mlruns folder
# Based on our discovery, it's inside experiment '1' and 'models' subdir
MODEL_SOURCE_PATH = os.path.abspath("mlruns/1/models/m-ff7c62d0386d4e0badf93c5dd6bb405d/artifacts")

def init_registry():
    client = MlflowClient()
    
    print(f"Initializing local MLflow registry for model: {MODEL_NAME}...")
    
    try:
        # 1. Create the registered model entry in the local database
        # This is the 'Library Catalog' entry
        client.create_registered_model(MODEL_NAME)
        print(f"Created registered model entry: {MODEL_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Model '{MODEL_NAME}' already exists in registry.")
        else:
            print(f"Note: {e}")

    # 2. Create Model Version 1
    # We point it to the actual folder on your disk
    print(f"Linking Version 1 to: {MODEL_SOURCE_PATH}")
    
    try:
        version = client.create_model_version(
            name=MODEL_NAME,
            source=MODEL_SOURCE_PATH,
            run_id=RUN_ID
        )
        print(f"\n✅ SUCCESS!")
        print(f"Model '{MODEL_NAME}' is now Version {version.version} in your local MLflow.")
        print(f"Current Stage: {version.current_stage}")
    except Exception as e:
        print(f"❌ Error creating version: {e}")

if __name__ == "__main__":
    init_registry()
