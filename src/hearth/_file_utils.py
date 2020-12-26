import json


def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj
