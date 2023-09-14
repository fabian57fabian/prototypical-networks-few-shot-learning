import os

import numpy as np
import torch

def load_centroids(centroids_patrh:str) -> (torch.tensor, list):
    files = [f for f in os.listdir(centroids_patrh) if f.endswith(".npy")]
    if len(files) == 0: return None, None

    classes, centroids = [], []
    for file in files:
        path = os.path.join(centroids_patrh, file)
        c = np.load(path)
        classes.append(file[:-4])
        centroids.append(torch.from_numpy(c))
    centroids_tensor = torch.stack(centroids)
    return centroids_tensor, classes



def save_centroids(path:str, centroids: torch.tensor):
    np.save(path, centroids.to("cpu"))
