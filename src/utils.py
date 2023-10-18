import os
import re
import yaml

import torch

def download_file_from_url(url: str, dest_dir: str):
    """
    @param url: Web url to download
    @param dest_dir: Destination directory for file
    """
    os.system(f'wget -q "{url}" -P {dest_dir}')


def yaml_load(file: str) -> dict:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
    Returns:
        (dict): YAML data.
    """
    if not (file.endswith('yaml') or file.endswith('yml')):
        raise Exception("File is not yaml or yml")
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()
        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        return data

def get_torch_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device.startswith("cpu"):
        return torch.device(device)
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            print("WWARN: Unable to set device to GPU because not available. Fallback to 'cpu'")
            return torch.device("cpu")
    raise Exception(f"Unknown device {device}")
