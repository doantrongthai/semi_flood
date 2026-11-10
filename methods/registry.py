from typing import Callable, Dict, Type

_METHOD_REGISTRY: Dict[str, Type] = {}


def register_method(name: str):
    """
    Decorator to register a new semi-supervised or supervised segmentation method.
    Usage:
      @register_method("fixmatch")
      class FixMatchMethod(BaseMethod): ...
    """
    def decorator(cls):
        key = name.lower()
        if key in _METHOD_REGISTRY:
            print(f"[Registry Warning] Overwriting existing method registration for '{key}'.")
        _METHOD_REGISTRY[key] = cls
        return cls
    return decorator


def get_method_class(name: str) -> Type:
    key = name.lower()
    if key not in _METHOD_REGISTRY:
        raise KeyError(
            f"Method '{name}' is not registered. Available methods: {list_available_methods()}"
        )
    return _METHOD_REGISTRY[key]


def list_available_methods() -> list:
    return sorted(list(_METHOD_REGISTRY.keys()))
