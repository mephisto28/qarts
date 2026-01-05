
_datasets = {}


def register_dataset(name: str):
    def wrapper(class_name: type):
        _datasets[name] = class_name
        return class_name
    return wrapper


def get_dataset(name: str) -> type:
    if name not in _datasets:
        raise ValueError(f"Dataset {name} not found")
    return _datasets[name]
