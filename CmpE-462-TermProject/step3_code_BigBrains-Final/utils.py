import os
from typing import Any, List
import pickle


def get_abs_path(file_or_dir: str) -> str:
    return os.path.join(os.path.dirname(__file__), file_or_dir)


def create_dir(dir_name: str) -> None:
    if not os.path.exists(get_abs_path(dir_name)):
        os.makedirs(get_abs_path(dir_name))


def print_list(list: List) -> None:
    for item in list:
        print(item)


def pickle_object(object, location) -> None:
    with open(location, 'wb') as file:
        pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_object(location) -> Any:
    with open(location, 'rb') as file:
        obj = pickle.load(file)
    return obj


def file_exists(file_name) -> bool:
    return os.path.isfile(file_name)
