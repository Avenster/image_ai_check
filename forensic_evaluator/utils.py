import torch
import yaml
import logging

def load_config(path='configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_device(device_str):
    if device_str == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
