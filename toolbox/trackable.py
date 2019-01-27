from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, TypeVar, cast


class Trackable(ABC):
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the object to a serializable dictionary
        Returns:
            The dictionary represents the object state
        """
        return {'__class__': self.__class__}

    def serialize_plain_attrs(self, state: Dict[str, Any],
                              attrs: List[str]) -> Dict[str, Any]:
        """
        Serialize non trackable attributes in a class
        Args:
            state: The current state dictionary
            attrs: The name of attributes

        Returns:
            state dictionary added the attributes
        """
        for attr in attrs:
            assert hasattr(self, attr)
            state[attr] = getattr(self, attr)
        return state

    def serialize_trackable_attrs(self, state: Dict[str, Any],
                                  attrs: List[str]) -> Dict[str, Any]:
        """
        Serialize trackable attributes in a class
        Args:
            state: The current state dictionary
            attrs: The name of the attributes

        Returns:
            state dictionary added with the state of the trackable attrs
        """
        for attr in attrs:
            assert hasattr(self, attr)
            obj = getattr(self, attr)
            assert isinstance(obj, Trackable)
            state[attr] = obj.serialize()
        return state

    @staticmethod
    @abstractmethod
    def deserialize(state: Dict[str, Any], strict: bool = False) -> 'Trackable':
        """
        Load the state into the object
        Args:
            state: The state dict
            strict: Whether throw exception if the state doesn't fully match up
        """


def serialize_list(lst: List[Trackable]) -> List[Dict[str, Any]]:
    return list(map(lambda t: t.serialize(), lst))


T = TypeVar('T', bound=Trackable)


def deserialize_state(state: Dict[str, Any],
                      cast_to: Type[T] = Trackable) -> T:
    assert '__class__' in state
    cls = state['__class__']
    assert issubclass(cls, Trackable)
    obj = cls.deserialize(state)
    return cast(cast_to, obj)


def deserialize_list(lst: List[Dict[str, Any]],
                     cast_to: Type[T] = Trackable) -> List[T]:
    return list(map(lambda state: deserialize_state(state, cast_to), lst))
