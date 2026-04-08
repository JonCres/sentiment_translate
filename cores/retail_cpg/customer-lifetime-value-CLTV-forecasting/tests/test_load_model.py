
import cloudpickle
import os

model_path = "/Users/pablo.andres/Workspace/ai_cores/github/ai_cores/cores/media_entertainment/predictive-CLTV-insights/data/06_models/bg_nbd_model.pickle"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    print(f"Model loaded: {type(model)}")
else:
    print("Model not found")
