import json


def read_json(json_path):
    # TODO To be documented and add typing in docstring and function definitin
    data = None

    with open(json_path) as f:
        data = json.load(f)

    return data
