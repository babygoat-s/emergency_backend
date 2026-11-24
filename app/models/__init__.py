from threading import Lock

# Serialize first-time PyTorch model construction. Inference runs independently.
torch_model_load_lock = Lock()
