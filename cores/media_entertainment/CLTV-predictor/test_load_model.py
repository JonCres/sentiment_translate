
import cloudpickle
import os

model_path = "data/06_models/bg_nbd_model.pickle"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    print(f"Model loaded: {type(model)}")
else:
    print(f"Model not found at: {os.path.abspath(model_path)}")
