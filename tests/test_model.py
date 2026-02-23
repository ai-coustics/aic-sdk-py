import os
from pathlib import Path

import pytest

import aic_sdk as aic


async def _download_model(model_id, download_dir, use_async):
    if use_async:
        return await aic.Model.download_async(model_id, download_dir)
    return aic.Model.download(model_id, download_dir)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_returns_path_string(tmp_path, use_async):
    model_path = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    assert isinstance(model_path, str)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_creates_model_file(tmp_path, use_async):
    model_path = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    assert os.path.exists(model_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_file_has_aicmodel_extension(tmp_path, use_async):
    model_path = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    assert model_path.endswith(".aicmodel")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_file_is_not_empty(tmp_path, use_async):
    model_path = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    assert os.path.getsize(model_path) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_downloaded_model_can_be_loaded(tmp_path, use_async):
    model_path = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    model = aic.Model.from_file(model_path)
    assert model is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_downloaded_model_has_correct_id(tmp_path, use_async):
    model_path = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    model = aic.Model.from_file(model_path)
    assert "sparrow" in model.get_id().lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_to_nested_directory(tmp_path, use_async):
    nested_dir = tmp_path / "models" / "nested"
    nested_dir.mkdir(parents=True)
    model_path = await _download_model("sparrow-s-48khz", str(nested_dir), use_async)
    assert os.path.exists(model_path)
    assert str(nested_dir) in model_path


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_same_model_twice_succeeds(tmp_path, use_async):
    model_path1 = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    model_path2 = await _download_model("sparrow-s-48khz", str(tmp_path), use_async)
    assert model_path1 == model_path2
    assert os.path.exists(model_path1)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_accepts_path_object(tmp_path, use_async):
    """Test that Model.download() and Model.download_async() accept pathlib.Path objects."""
    download_dir = Path(tmp_path)
    model_path = await _download_model("sparrow-s-48khz", download_dir, use_async)
    assert isinstance(model_path, str)
    assert os.path.exists(model_path)
    assert model_path.endswith(".aicmodel")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_download_with_path_object_creates_file(tmp_path, use_async):
    """Test that downloading with a Path object creates the expected model file."""
    download_dir = Path(tmp_path) / "models"
    download_dir.mkdir()
    model_path = await _download_model("sparrow-s-48khz", download_dir, use_async)
    assert os.path.exists(model_path)
    assert os.path.getsize(model_path) > 0
    # Verify the model can be loaded
    model = aic.Model.from_file(model_path)
    assert model is not None
    assert "sparrow" in model.get_id().lower()


def test_from_file_accepts_path_object(tmp_path):
    """Test that Model.from_file() accepts pathlib.Path objects."""
    # First download a model using a Path object
    download_dir = Path(tmp_path)
    model_path_str = aic.Model.download("sparrow-s-48khz", download_dir)

    # Now load it using a Path object
    model_path = Path(model_path_str)
    model = aic.Model.from_file(model_path)
    assert model is not None
    assert "sparrow" in model.get_id().lower()


def test_from_file_accepts_path_object_directly(tmp_path):
    """Test that Model.from_file() accepts pathlib.Path objects directly without conversion."""
    # Download model
    download_dir = Path(tmp_path)
    model_path_str = aic.Model.download("sparrow-s-48khz", download_dir)
    model_path = Path(model_path_str)

    # Load using Path object directly
    model1 = aic.Model.from_file(model_path)
    # Also verify it works with string for comparison
    model2 = aic.Model.from_file(model_path_str)

    # Both should work and produce the same model ID
    assert model1.get_id() == model2.get_id()
