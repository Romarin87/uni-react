"""Unit tests for the Registry class."""
import pytest

from uni_react.registry import Registry


def test_register_and_get():
    reg = Registry("test")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo


def test_register_duplicate_raises():
    reg = Registry("test")

    @reg.register("bar")
    class Bar:
        pass

    with pytest.raises(KeyError, match="already registered"):
        @reg.register("bar")
        class Bar2:
            pass


def test_build_from_dict():
    reg = Registry("test")

    @reg.register("widget")
    class Widget:
        def __init__(self, size: int = 1):
            self.size = size

    obj = reg.build({"type": "widget", "size": 42})
    assert isinstance(obj, Widget)
    assert obj.size == 42


def test_build_missing_type_raises():
    reg = Registry("test")
    with pytest.raises(ValueError, match="'type' key is missing"):
        reg.build({"size": 1})


def test_get_unknown_raises():
    reg = Registry("test")
    with pytest.raises(KeyError, match="not found"):
        reg.get("nonexistent")


def test_list_aliases():
    reg = Registry("test")

    @reg.register("a")
    class A:
        pass

    @reg.register("b")
    class B:
        pass

    assert reg.list_aliases() == ["a", "b"]


def test_contains():
    reg = Registry("test")

    @reg.register("x")
    class X:
        pass

    assert "x" in reg
    assert "y" not in reg


def test_register_or_replace():
    reg = Registry("test")

    @reg.register("item")
    class V1:
        pass

    @reg.register_or_replace("item")
    class V2:
        pass

    assert reg.get("item") is V2
