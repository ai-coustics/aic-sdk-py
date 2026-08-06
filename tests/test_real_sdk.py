import re

import numpy as np
from conftest import chunks, make_sine_noise

import aic_sdk as aic


def test_real_sdk_processing_changes_signal(processor):
    config = aic.ProcessorConfig(48000, 480, False)
    processor.initialize(config)
    proc_ctx = processor.get_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)

    audio = make_sine_noise(4800)
    original = audio.copy()

    # process in chunks
    for s, e in chunks(audio.shape[0], 480):
        chunk = audio[s:e]
        if chunk.shape[0] < 480:
            padded = np.zeros(480, dtype=audio.dtype)
            padded[: chunk.shape[0]] = chunk
            processor.process(padded)
            audio[s:e] = padded[: chunk.shape[0]]
        else:
            processor.process(chunk)

    assert audio.shape == original.shape

    # Ensure the model altered the signal (not identical to input)
    # assert not np.allclose(audio, original)  # TODO: enable this line.

    # Ensure finite values within a reasonable bound
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 5.0


def test_real_sdk_processing_submit_future(processor):
    config = aic.ProcessorConfig(48000, 480, False)
    processor.initialize(config)
    proc_ctx = processor.get_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)

    audio = make_sine_noise(4800)
    original = audio.copy()

    for s, e in chunks(audio.shape[0], 480):
        chunk = audio[s:e]
        if chunk.shape[0] < 480:
            padded = np.zeros(480, dtype=audio.dtype)
            padded[: chunk.shape[0]] = chunk
            processed = processor.process(padded)
            audio[s:e] = processed[: chunk.shape[0]]
        else:
            processed = processor.process(chunk)
            audio[s:e] = processed

    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_initialize_uses_optimal_block_size(processor, model):
    # Get optimal configuration (includes optimal block size)
    optimal_config = aic.ProcessorConfig.optimal(model)
    sr = optimal_config.sample_rate
    frames = optimal_config.block_size

    # Initialize with optimal config
    processor.initialize(optimal_config)

    # Sanity check processing end-to-end
    proc_ctx = processor.get_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.9)
    audio = make_sine_noise(frames * 8, sr=sr)
    original = audio.copy()

    for s, e in chunks(audio.shape[0], frames):
        chunk = audio[s:e]
        if chunk.shape[0] < frames:
            padded = np.zeros(frames, dtype=audio.dtype)
            padded[: chunk.shape[0]] = chunk
            processed = processor.process(padded)
            audio[s:e] = processed[: chunk.shape[0]]
        else:
            processed = processor.process(chunk)
            audio[s:e] = processed

    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_vad_detection_runs(vad_model, license_key):
    """Test voice activity detection with a dedicated VAD model."""
    config = aic.ProcessorConfig.optimal(vad_model)
    vad = aic.Vad(vad_model, license_key, config)
    context = vad.get_context()

    context.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.05)
    context.set_parameter(aic.VadParameter.Sensitivity, 0.5)
    assert isinstance(context.get_parameter(aic.VadParameter.SpeechHoldDuration), float)
    assert 0.0 <= context.get_parameter(aic.VadParameter.Sensitivity) <= 1.0

    audio = make_sine_noise(config.block_size * 10, sr=config.sample_rate)
    prediction = None
    for start in range(0, audio.shape[0], config.block_size):
        vad.process(audio[start : start + config.block_size])
        prediction = context.is_speech_detected()

    assert isinstance(prediction, bool)


def test_real_sdk_model_processing(model, license_key):
    model_id = "rook-s-48khz"
    model_path = aic.Model.download(model_id, "./models")
    model = aic.Model.from_file(model_path)
    khz_pattern = re.compile(r"-(\d+)khz\b")
    block_size = 480
    variable_block_size = False

    # L16/S16 use 16k, L8/S8 use 8k
    probe_sr = int(khz_pattern.search(model_id).group(1)) * 1000

    # initial processor
    config_initial = aic.ProcessorConfig(probe_sr, block_size, variable_block_size)
    processor_initial = aic.Processor(model, license_key)
    processor_initial.initialize(config_initial)
    optimal_sr = model.get_optimal_sample_rate()
    optimal_block_size = model.get_optimal_block_size(optimal_sr)

    config = aic.ProcessorConfig(optimal_sr, optimal_block_size, variable_block_size)
    processor = aic.Processor(model, license_key)
    processor.initialize(config)
    proc_ctx = processor.get_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    audio = make_sine_noise(optimal_block_size * 10, sr=optimal_sr)
    original = audio.copy()

    for s, e in chunks(audio.shape[0], optimal_block_size):
        chunk = audio[s:e]
        if chunk.shape[0] < optimal_block_size:
            padded = np.zeros(optimal_block_size, dtype=audio.dtype)
            padded[: chunk.shape[0]] = chunk
            processed = processor.process(padded)
            audio[s:e] = processed[: chunk.shape[0]]
        else:
            processed = processor.process(chunk)
            audio[s:e] = processed

    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_auto_model_selection(model, license_key):
    sample_rate = 16000
    block_size = 240
    processor = aic.Processor(model, license_key)
    config = aic.ProcessorConfig(sample_rate, block_size, False)
    processor.initialize(config)
    optimal_sr = model.get_optimal_sample_rate()
    optimal_block_size = model.get_optimal_block_size(optimal_sr)
    assert optimal_sr == sample_rate
    assert optimal_block_size == block_size
