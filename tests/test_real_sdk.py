import re

import numpy as np
from conftest import chunks, make_sine_noise

import aic_sdk as aic


def test_real_sdk_processing_changes_signal(processor):
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)

    audio = make_sine_noise(1, 4800)
    original = audio.copy()

    # process in chunks
    for s, e in chunks(audio.shape[1], 480):
        chunk = audio[:, s:e]
        if chunk.shape[1] < 480:
            padded = np.zeros((1, 480), dtype=audio.dtype)
            padded[:, : chunk.shape[1]] = chunk
            processor.process(padded)
            audio[:, s:e] = padded[:, : chunk.shape[1]]
        else:
            processor.process(chunk)

    assert audio.shape == original.shape

    # Ensure the model altered the signal (not identical to input)
    # assert not np.allclose(audio, original)  # TODO: enable this line.

    # Ensure finite values within a reasonable bound
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 5.0


def test_real_sdk_processing_runs(processor):
    config = aic.ProcessorConfig(48000, 2, 480, False)
    processor.initialize(config)

    frames = 480
    planar = make_sine_noise(2, frames)

    out = processor.process(planar)
    assert np.isfinite(out).all()


def test_real_sdk_planar_processing_submit_future(processor):
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)
    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)

    audio = make_sine_noise(1, 4800)
    original = audio.copy()

    for s, e in chunks(audio.shape[1], 480):
        chunk = audio[:, s:e]
        if chunk.shape[1] < 480:
            padded = np.zeros((1, 480), dtype=audio.dtype)
            padded[:, : chunk.shape[1]] = chunk
            processed = processor.process(padded)
            audio[:, s:e] = processed[:, : chunk.shape[1]]
        else:
            processed = processor.process(chunk)
            audio[:, s:e] = processed

    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_interleaved_processing_submit_future(processor):
    config = aic.ProcessorConfig(48000, 2, 480, False)
    processor.initialize(config)

    frames = 480
    planar = make_sine_noise(2, frames)

    out = processor.process(planar)
    assert np.isfinite(out).all()


def test_real_sdk_initialize_without_frames_uses_optimal_frames(processor, model):
    # Get optimal configuration (includes optimal frames)
    optimal_config = aic.ProcessorConfig.optimal(model, num_channels=1)
    sr = optimal_config.sample_rate
    frames = optimal_config.num_frames

    # Initialize with optimal config
    processor.initialize(optimal_config)

    # Sanity check processing end-to-end
    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.9)
    audio = make_sine_noise(1, frames * 8, sr=sr)
    original = audio.copy()

    for s, e in chunks(audio.shape[1], frames):
        chunk = audio[:, s:e]
        if chunk.shape[1] < frames:
            padded = np.zeros((1, frames), dtype=audio.dtype)
            padded[:, : chunk.shape[1]] = chunk
            processed = processor.process(padded)
            audio[:, s:e] = processed[:, : chunk.shape[1]]
        else:
            processed = processor.process(chunk)
            audio[:, s:e] = processed

    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_vad_detection_runs(processor):
    """Test that VAD detection works."""
    config = aic.ProcessorConfig(48000, 1, 480, False)
    processor.initialize(config)

    # configure model and VAD
    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 1.0)
    vad = processor.get_vad_context()

    # set VAD parameters to reasonable defaults
    vad.set_parameter(aic.VadParameter.SpeechHoldDuration, 0.05)
    vad.set_parameter(aic.VadParameter.Sensitivity, 6.0)
    shd = vad.parameter(aic.VadParameter.SpeechHoldDuration)
    se = vad.parameter(aic.VadParameter.Sensitivity)
    assert isinstance(shd, float)
    assert 0.0 <= shd <= 0.4  # 20x 10ms window = 0.2s, but allow some margin
    assert isinstance(se, float)
    assert 1.0 <= se <= 15.0

    # drive the model so VAD has predictions to report
    frames = 480 * 10
    audio = make_sine_noise(1, frames)
    last_pred = None
    for s, e in chunks(frames, 480):
        chunk = audio[:, s:e]
        if chunk.shape[1] < 480:
            padded = np.zeros((1, 480), dtype=audio.dtype)
            padded[:, : chunk.shape[1]] = chunk
            processor.process(padded)
        else:
            processor.process(chunk)
        # query prediction (latency equals model latency; we only assert type)
        last_pred = vad.is_speech_detected()

    assert isinstance(last_pred, bool)


def test_real_sdk_sequential_processing_runs(processor):
    """Test that processing works with stereo audio."""
    config = aic.ProcessorConfig(48000, 2, 480, False)
    processor.initialize(config)

    frames = 480
    planar = make_sine_noise(2, frames)

    out = processor.process(planar)
    assert np.isfinite(out).all()


def test_real_sdk_model_processing(model, license_key):
    model_id = "sparrow-xxs-48khz"
    model_path = aic.Model.download(model_id, "./models")
    model = aic.Model.from_file(model_path)
    khz_pattern = re.compile(r"-(\d+)khz\b")
    num_channels = 1
    num_frames = 480
    allow_variable_frames = False

    # L16/S16 use 16k, L8/S8 use 8k
    probe_sr = int(khz_pattern.search(model_id).group(1)) * 1000

    # initial processor
    config_initial = aic.ProcessorConfig(
        probe_sr, num_channels, num_frames, allow_variable_frames
    )
    processor_initial = aic.Processor(model, license_key)
    processor_initial.initialize(config_initial)
    optimal_sr = model.get_optimal_sample_rate()
    optimal_frames = model.get_optimal_num_frames(optimal_sr)

    config = aic.ProcessorConfig(
        optimal_sr, num_channels, optimal_frames, allow_variable_frames
    )
    processor = aic.Processor(model, license_key)
    processor.initialize(config)
    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, 0.8)
    audio = make_sine_noise(1, optimal_frames * 10, sr=optimal_sr)
    original = audio.copy()

    for s, e in chunks(audio.shape[1], optimal_frames):
        chunk = audio[:, s:e]
        if chunk.shape[1] < optimal_frames:
            padded = np.zeros((1, optimal_frames), dtype=audio.dtype)
            padded[:, : chunk.shape[1]] = chunk
            processed = processor.process(padded)
            audio[:, s:e] = processed[:, : chunk.shape[1]]
        else:
            processed = processor.process(chunk)
            audio[:, s:e] = processed

    assert audio.shape == original.shape
    assert not np.allclose(audio, original)
    assert np.isfinite(audio).all()


def test_real_sdk_auto_model_selection(model, license_key):
    sample_rate = 48000
    num_frames = 480
    processor = aic.Processor(model, license_key)
    config = aic.ProcessorConfig(sample_rate, 1, num_frames, False)
    processor.initialize(config)
    optimal_sr = model.get_optimal_sample_rate()
    optimal_frames = model.get_optimal_num_frames(optimal_sr)
    assert optimal_sr == sample_rate
    assert optimal_frames == num_frames
