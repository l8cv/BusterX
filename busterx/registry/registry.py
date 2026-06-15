# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import warnings
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            warnings.warn(
                f"[BusterX] An object named '{name}' was already registered in '{self._name}' registry! We will NOT overwrite the existing one.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._obj_map[name] = obj

    def register(self, obj: Any = None, name: Optional[str] = None) -> Any:
        """
        Register the given object under the name `obj.__name__` or the given name.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                _name = name
                if _name is None:
                    _name = func_or_class.__name__
                self._do_register(_name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"[BusterX] No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        return "Registry of {}\n".format(self._name)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__
