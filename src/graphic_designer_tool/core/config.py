""" Config core functions """
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict[str, Any]: A dictionary containing the configuration settings.
                        The dictionary keys are strings, and the values can
                        be of any type.

    Raises:
        FileNotFoundError: If the specified config_path does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    return config
