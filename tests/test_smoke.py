from importlib.metadata import version

import slmstrict


def test_package_is_importable() -> None:
    assert slmstrict.__name__ == "slmstrict"


def test_version_is_exposed() -> None:
    assert isinstance(slmstrict.__version__, str)
    assert slmstrict.__version__


def test_version_matches_installed_metadata() -> None:
    assert slmstrict.__version__ == version("slmstrict")
