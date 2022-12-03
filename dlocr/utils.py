import json


def load_config(config_path):
    with open(config_path, "r") as infile:
        return dict(json.load(infile))
