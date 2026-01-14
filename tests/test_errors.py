import os

import numpy as np
import pytest

import aic_sdk as aic
from conftest import create_processor_or_skip, create_processor_async_or_skip


def test_empty_license_key_raises_license_format_invalid_error(model):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.Processor(model, "")
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)


def test_invalid_license_key_raises_license_format_invalid_error(model):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.Processor(model, "not-a-valid-license-key")
    assert exc_info.value.message
    assert len(exc_info.value.message) > 0


def test_whitespace_license_key_raises_license_format_invalid_error(model):
    with pytest.raises(aic.LicenseFormatInvalidError):
        aic.Processor(model, "   ")


@pytest.mark.asyncio
async def test_invalid_license_key_raises_license_format_invalid_error_async(model):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.ProcessorAsync(model, "invalid-key")
    assert exc_info.value.message


def test_nonexistent_model_file_raises_filesystem_error():
    with pytest.raises(aic.FileSystemError) as exc_info:
        aic.Model.from_file("/nonexistent/path/to/model.aicmodel")
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)


def test_directory_path_raises_filesystem_error(tmp_path):
    with pytest.raises(aic.FileSystemError):
        aic.Model.from_file(str(tmp_path))


def test_empty_file_raises_model_invalid_or_unaligned_error(tmp_path):
    fake_model = tmp_path / "empty.aicmodel"
    fake_model.write_bytes(b"")
    with pytest.raises((aic.ModelInvalidError, aic.ModelDataUnalignedError)):
        aic.Model.from_file(str(fake_model))


def test_random_bytes_file_raises_model_invalid_or_unaligned_error(tmp_path):
    fake_model = tmp_path / "random.aicmodel"
    fake_model.write_bytes(os.urandom(1024))
    with pytest.raises((aic.ModelInvalidError, aic.ModelDataUnalignedError)):
        aic.Model.from_file(str(fake_model))


def test_text_file_raises_model_invalid_or_unaligned_error(tmp_path):
    fake_model = tmp_path / "text.aicmodel"
    fake_model.write_text("This is not a model file")
    with pytest.raises((aic.ModelInvalidError, aic.ModelDataUnalignedError)):
        aic.Model.from_file(str(fake_model))


def test_empty_path_raises_error():
    with pytest.raises(
        (aic.ModelFilePathInvalidError, aic.FileSystemError, ValueError)
    ):
        aic.Model.from_file("")


def test_null_byte_in_path_raises_error():
    with pytest.raises(
        (aic.ModelFilePathInvalidError, aic.FileSystemError, ValueError, BaseException)
    ):
        aic.Model.from_file("/path/with\x00null/model.aicmodel")


def test_nonexistent_model_id_raises_model_download_error(tmp_path):
    with pytest.raises(aic.ModelDownloadError) as exc_info:
        aic.Model.download("nonexistent-model-id-12345", str(tmp_path))
    assert exc_info.value.message
    assert isinstance(exc_info.value.message, str)
    assert exc_info.value.details
    assert isinstance(exc_info.value.details, str)


@pytest.mark.asyncio
async def test_nonexistent_model_id_raises_model_download_error_async(tmp_path):
    with pytest.raises(aic.ModelDownloadError) as exc_info:
        await aic.Model.download_async("nonexistent-model-id-12345", str(tmp_path))
    assert exc_info.value.message
    assert exc_info.value.details


def test_process_before_initialize_raises_model_not_initialized_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    audio = np.zeros((1, 480), dtype=np.float32)
    with pytest.raises(aic.ModelNotInitializedError) as exc_info:
        processor.process(audio)
    assert exc_info.value.message


def test_reset_before_initialize_may_raise_model_not_initialized_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    ctx = processor.get_processor_context()
    try:
        ctx.reset()
    except aic.ModelNotInitializedError:
        pass


def test_wrong_channel_count_raises_audio_config_mismatch_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    audio = np.zeros((2, 480), dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError) as exc_info:
        processor.process(audio)
    assert exc_info.value.message


def test_smaller_frame_count_raises_audio_config_mismatch_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, allow_variable_frames=False)
    processor.initialize(config)
    audio = np.zeros((1, 240), dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError) as exc_info:
        processor.process(audio)
    assert exc_info.value.message


def test_larger_frame_count_raises_audio_config_mismatch_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, allow_variable_frames=False)
    processor.initialize(config)
    audio = np.zeros((1, 960), dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError) as exc_info:
        processor.process(audio)
    assert exc_info.value.message


@pytest.mark.asyncio
async def test_wrong_channel_count_raises_audio_config_mismatch_error_async(
    model, license_key
):
    processor = create_processor_async_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    await processor.initialize_async(config)
    audio = np.zeros((2, 480), dtype=np.float32)
    with pytest.raises(aic.AudioConfigMismatchError):
        await processor.process_async(audio)


def test_unsupported_sample_rate_raises_audio_config_unsupported_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(1, 1, 480, False)
    with pytest.raises(aic.AudioConfigUnsupportedError) as exc_info:
        processor.initialize(config)
    assert exc_info.value.message


def test_zero_channels_raises_audio_config_unsupported_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 0, 480, False)
    with pytest.raises(aic.AudioConfigUnsupportedError) as exc_info:
        processor.initialize(config)
    assert exc_info.value.message


def test_zero_frames_raises_audio_config_unsupported_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 0, False)
    with pytest.raises(aic.AudioConfigUnsupportedError) as exc_info:
        processor.initialize(config)
    assert exc_info.value.message


def test_enhancement_level_above_max_raises_parameter_out_of_range_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 5.0)
    assert exc_info.value.message


def test_enhancement_level_negative_raises_parameter_out_of_range_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, -0.5)
    assert exc_info.value.message


def test_voice_gain_above_max_raises_parameter_out_of_range_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 10.0)
    assert exc_info.value.message


def test_voice_gain_below_min_raises_parameter_out_of_range_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        ctx.set_parameter(aic.ProcessorParameter.VoiceGain, 0.01)
    assert exc_info.value.message


def test_bypass_above_max_raises_parameter_out_of_range_error(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        ctx.set_parameter(aic.ProcessorParameter.Bypass, 2.0)
    assert exc_info.value.message


def test_vad_sensitivity_above_max_raises_parameter_out_of_range_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        vad.set_parameter(aic.VadParameter.Sensitivity, 100.0)
    assert exc_info.value.message


def test_vad_sensitivity_below_min_raises_parameter_out_of_range_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        vad.set_parameter(aic.VadParameter.Sensitivity, 0.1)
    assert exc_info.value.message


def test_vad_speech_hold_duration_negative_raises_parameter_out_of_range_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        vad.set_parameter(aic.VadParameter.SpeechHoldDuration, -0.1)
    assert exc_info.value.message


def test_vad_minimum_speech_duration_above_max_raises_parameter_out_of_range_error(
    model, license_key
):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    vad = processor.get_vad_context()
    with pytest.raises(aic.ParameterOutOfRangeError) as exc_info:
        vad.set_parameter(aic.VadParameter.MinimumSpeechDuration, 5.0)
    assert exc_info.value.message


@pytest.mark.parametrize(
    "error_class",
    [
        aic.ParameterOutOfRangeError,
        aic.ModelNotInitializedError,
        aic.AudioConfigUnsupportedError,
        aic.AudioConfigMismatchError,
        aic.EnhancementNotAllowedError,
        aic.InternalError,
        aic.ParameterFixedError,
        aic.LicenseFormatInvalidError,
        aic.LicenseVersionUnsupportedError,
        aic.LicenseExpiredError,
        aic.ModelInvalidError,
        aic.ModelVersionUnsupportedError,
        aic.ModelFilePathInvalidError,
        aic.FileSystemError,
        aic.ModelDataUnalignedError,
        aic.ModelDownloadError,
        aic.UnknownError,
    ],
)
def test_error_is_exception_subclass(error_class):
    assert issubclass(error_class, Exception)


@pytest.mark.parametrize(
    "error_class",
    [
        aic.ParameterOutOfRangeError,
        aic.ModelNotInitializedError,
        aic.AudioConfigUnsupportedError,
        aic.AudioConfigMismatchError,
        aic.EnhancementNotAllowedError,
        aic.InternalError,
        aic.ParameterFixedError,
        aic.LicenseFormatInvalidError,
        aic.LicenseVersionUnsupportedError,
        aic.LicenseExpiredError,
        aic.ModelInvalidError,
        aic.ModelVersionUnsupportedError,
        aic.ModelFilePathInvalidError,
        aic.FileSystemError,
        aic.ModelDataUnalignedError,
        aic.ModelDownloadError,
        aic.UnknownError,
    ],
)
def test_error_is_base_exception_subclass(error_class):
    assert issubclass(error_class, BaseException)


def test_license_format_invalid_error_has_message_attribute(model):
    try:
        aic.Processor(model, "invalid")
    except aic.LicenseFormatInvalidError as e:
        assert hasattr(e, "message")
        assert isinstance(e.message, str)
        assert len(e.message) > 0


def test_model_download_error_has_message_and_details_attributes(tmp_path):
    try:
        aic.Model.download("nonexistent-model", str(tmp_path))
    except aic.ModelDownloadError as e:
        assert hasattr(e, "message")
        assert hasattr(e, "details")
        assert isinstance(e.message, str)
        assert isinstance(e.details, str)


def test_filesystem_error_has_message_attribute():
    try:
        aic.Model.from_file("/nonexistent/model.aicmodel")
    except aic.FileSystemError as e:
        assert hasattr(e, "message")
        assert isinstance(e.message, str)


def test_license_format_invalid_error_message_is_descriptive(model):
    try:
        aic.Processor(model, "bad-key")
    except aic.LicenseFormatInvalidError as e:
        assert len(e.message) > 10
        assert len(str(e)) > 0


def test_audio_config_unsupported_error_message_is_descriptive(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 0, 480, False)
    try:
        processor.initialize(config)
    except aic.AudioConfigUnsupportedError as e:
        assert len(e.message) > 10


def test_parameter_out_of_range_error_message_is_descriptive(model, license_key):
    processor = create_processor_or_skip(model, license_key)
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    ctx = processor.get_processor_context()
    try:
        ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 999.0)
    except aic.ParameterOutOfRangeError as e:
        assert len(e.message) > 10
