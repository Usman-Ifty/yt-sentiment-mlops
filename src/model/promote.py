import mlflow
import json
import os
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "yt-sentiment-bert")
THRESHOLD  = 0.80   # minimum F1 to allow promotion

def promote_model():
    client = MlflowClient()

    # get all versions of this model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        print("No model versions found.")
        return

    # latest version is the one just trained
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    latest_version = latest.version
    run_id = latest.run_id

    # fetch metrics from that run
    run = client.get_run(run_id)
    test_f1 = run.data.metrics.get("test_f1", 0)

    print(f"Latest model version : {latest_version}")
    print(f"Test F1              : {test_f1:.4f}")
    print(f"Threshold            : {THRESHOLD}")

    # check if there's already a production model
    prod_versions = [
        v for v in versions if v.current_stage == "Production"
    ]

    if prod_versions:
        prod = prod_versions[0]
        prod_run = client.get_run(prod.run_id)
        prod_f1  = prod_run.data.metrics.get("test_f1", 0)
        print(f"Current production F1: {prod_f1:.4f}")

        if test_f1 <= prod_f1:
            print("New model is NOT better than production. Skipping promotion.")
            return

    if test_f1 < THRESHOLD:
        print(f"F1 {test_f1:.4f} is below threshold {THRESHOLD}. Not promoting.")
        return

    # archive old production model
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived",
        )
        print(f"Archived version {v.version}")

    # promote new model
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production",
    )

    print(f"\nPromoted version {latest_version} to Production!")

    # save the production version number for Docker build
    os.makedirs("models", exist_ok=True)
    with open("models/production_version.json", "w") as f:
        json.dump({
            "version"  : latest_version,
            "run_id"   : run_id,
            "test_f1"  : test_f1,
        }, f, indent=2)


if __name__ == "__main__":
    promote_model()
