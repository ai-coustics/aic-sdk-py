import aic_sdk as aic


def test_get_sdk_version_returns_string():
    version = aic.get_sdk_version()
    assert isinstance(version, str)


def test_get_sdk_version_is_not_empty():
    version = aic.get_sdk_version()
    assert len(version) > 0


def test_get_sdk_version_has_version_format():
    version = aic.get_sdk_version()
    parts = version.split(".")
    assert len(parts) >= 2


def test_get_compatible_model_version_returns_int():
    version = aic.get_compatible_model_version()
    assert isinstance(version, int)


def test_get_compatible_model_version_is_positive():
    version = aic.get_compatible_model_version()
    assert version > 0
