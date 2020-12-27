import json
import pathlib


def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path: str):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def mkdirs_if_not_exist(path, verbose: bool = False):
    path = pathlib.Path(path)
    if not path.exists():
        if verbose:
            print(f'directory {path} does not exist. creating...')
        path.mkdir(parents=True, exist_ok=True)
