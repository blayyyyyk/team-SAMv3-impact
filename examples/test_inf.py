from impact_team_2.inference import predict
import urllib.request
from pathlib import Path
import numpy as np

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

images = np.load(DATASETS_DIR / "images.npz")
masks = np.load(DATASETS_DIR / "masks.npz")

print(images.files)  # see available keys
print(masks.files)

    

# predict()
