import numpy as np
import pytest

import aic_sdk as aic

_SCORE_ATTRS = (
    "risk_score",
    "speaker_reverb",
    "speaker_loudness",
    "interfering_speech",
    "media_speech",
    "noise",
    "packet_loss",
)


def assert_scores_in_range(result):
    for attr in _SCORE_ATTRS:
        score = getattr(result, attr)
        assert 0.0 <= score <= 1.0, f"{attr}={score} out of range"


def make_pair_or_skip(model, license_key):
    try:
        return aic.analyzer_pair(model, license_key)
    except aic.LicenseVersionUnsupportedError:
        pytest.skip("License version incompatible with SDK version")
    except aic.LicenseExpiredError:
        pytest.skip("License has expired")


@pytest.mark.parametrize(
    "license_key",
    ["", "invalid-license-key"],
)
def test_analyzer_pair_requires_valid_license_key(analysis_model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.analyzer_pair(analysis_model, license_key)

    assert "License key format is invalid or corrupted" in str(exc_info.value)


def test_collector_buffer_and_analyzer_returns_scores(analysis_model, license_key):
    collector, analyzer = make_pair_or_skip(analysis_model, license_key)

    config = aic.ProcessorConfig.optimal(analysis_model)
    collector.initialize(config)

    audio = np.zeros(config.block_size, dtype=np.float32)
    collector.buffer(audio)

    result = analyzer.analyze_buffered()

    assert isinstance(result, aic.AnalysisResult)
    assert_scores_in_range(result)


def test_collector_buffer_accepts_reversed_view(analysis_model, license_key):
    """A reversed (negative-stride) 1D view must be normalized, not passed through raw.

    ndarray's to_owned() preserves contiguous-but-reversed strides, which would make
    as_slice() return None again; buffer() must use as_standard_layout() to avoid that.
    """
    collector, analyzer = make_pair_or_skip(analysis_model, license_key)

    config = aic.ProcessorConfig.optimal(analysis_model)
    collector.initialize(config)

    audio = np.arange(config.block_size, dtype=np.float32)[::-1]
    assert audio.strides[0] < 0  # sanity check: this really is a reversed view

    collector.buffer(audio)
    result = analyzer.analyze_buffered()

    assert isinstance(result, aic.AnalysisResult)
    assert_scores_in_range(result)


def test_analyzer_reset_keeps_collector_initialized(analysis_model, license_key):
    collector, analyzer = make_pair_or_skip(analysis_model, license_key)

    config = aic.ProcessorConfig.optimal(analysis_model)
    collector.initialize(config)

    analyzer.reset()

    collector.buffer(np.zeros(config.block_size, dtype=np.float32))
    result = analyzer.analyze_buffered()

    assert_scores_in_range(result)


def test_analyzer_terminate_session_prevents_further_analysis(
    analysis_model, license_key
):
    collector, analyzer = make_pair_or_skip(analysis_model, license_key)
    collector.initialize(aic.ProcessorConfig.optimal(analysis_model))
    analyzer.terminate_session()

    with pytest.raises(aic.ProcessingNotAllowedError):
        analyzer.analyze_buffered()


def test_analyzer_pair_keeps_model_alive_after_model_drop(license_key):
    import gc

    # Create the model locally so the returned pair holds the only remaining reference once we
    # drop ours. The collector/analyzer must keep the native model state alive; otherwise the
    # analyze_buffered() below would use freed memory.
    model = aic.Model.from_file(aic.Model.download("tyto-l-16khz", "./models"))
    collector, analyzer = make_pair_or_skip(model, license_key)
    config = aic.ProcessorConfig.optimal(model)

    del model
    gc.collect()

    collector.initialize(config)
    collector.buffer(np.zeros(config.block_size, dtype=np.float32))
    result = analyzer.analyze_buffered()

    assert_scores_in_range(result)
