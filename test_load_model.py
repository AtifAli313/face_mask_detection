import pickle
import h5py
import numpy as np

# Try different methods to load the model
model_path = "mask_detector.model"

print("[INFO] Attempting to load model...")

# Method 1: Try pickle
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("[SUCCESS] Loaded as pickle file")
    print(f"Type: {type(model)}")
except Exception as e:
    print(f"[FAILED] Pickle: {e}")

# Method 2: Try HDF5
try:
    with h5py.File(model_path, 'r') as f:
        print("[SUCCESS] Valid HDF5 file")
        print(f"Keys: {list(f.keys())}")
except Exception as e:
    print(f"[FAILED] HDF5: {e}")

# Method 3: Check file header
print("\n[INFO] File header (first 20 bytes):")
with open(model_path, 'rb') as f:
    header = f.read(20)
    print(f"Hex: {header.hex()}")
    print(f"ASCII: {header}")
