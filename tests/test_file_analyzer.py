import aic_sdk as aic
import numpy as np
import pytest

ANALYSIS_WINDOW_SECONDS = 5

_SCORE_ATTRS = (
    "risk_score",
    "speaker_reverb",
    "speaker_loudness",
    "interfering_speech",
    "media_speech",
    "noise",
    "packet_loss",
)


def assert_all_scores_in_range(results):
    for result in results:
        for attr in _SCORE_ATTRS:
            score = getattr(result, attr)
            assert 0.0 <= score <= 1.0, f"{attr}={score} out of range"


def make_file_analyzer_or_skip(model, license_key):
    try:
        return aic.FileAnalyzer(model, license_key)
    except aic.LicenseVersionUnsupportedError:
        pytest.skip("License version incompatible with SDK version")
    except aic.LicenseExpiredError:
        pytest.skip("License has expired")


@pytest.mark.parametrize(
    "license_key",
    ["", "invalid-license-key"],
)
def test_file_analyzer_requires_valid_license_key(analysis_model, license_key):
    with pytest.raises(aic.LicenseFormatInvalidError) as exc_info:
        aic.FileAnalyzer(analysis_model, license_key)

    assert "License key format is invalid or corrupted" in str(exc_info.value)


def test_analyze_rejects_zero_sample_rate_or_step(analysis_model, license_key):
    analyzer = make_file_analyzer_or_skip(analysis_model, license_key)
    audio = np.zeros(16, dtype=np.float32)

    with pytest.raises(aic.AudioConfigUnsupportedError):
        analyzer.analyze(audio, 0, 160)

    with pytest.raises(aic.AudioConfigUnsupportedError):
        analyzer.analyze(audio, 16000, 0)


def test_analyze_short_audio_returns_single_padded_result(analysis_model, license_key):
    analyzer = make_file_analyzer_or_skip(analysis_model, license_key)
    sample_rate = analysis_model.get_optimal_sample_rate()
    step_samples = analysis_model.get_optimal_num_frames(sample_rate)
    audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second, shorter than the window

    results = analyzer.analyze(audio, sample_rate, step_samples)

    assert len(results) == 1
    assert_all_scores_in_range(results)


def test_analyze_exact_window_returns_single_result(analysis_model, license_key):
    analyzer = make_file_analyzer_or_skip(analysis_model, license_key)
    sample_rate = analysis_model.get_optimal_sample_rate()
    step_samples = analysis_model.get_optimal_num_frames(sample_rate)
    window_samples = sample_rate * ANALYSIS_WINDOW_SECONDS
    audio = np.zeros(window_samples, dtype=np.float32)

    results = analyzer.analyze(audio, sample_rate, step_samples)

    assert len(results) == 1
    assert_all_scores_in_range(results)


def test_analyze_defaults_step_to_analysis_window_size(analysis_model, license_key):
    analyzer = make_file_analyzer_or_skip(analysis_model, license_key)
    sample_rate = analysis_model.get_optimal_sample_rate()
    window_samples = sample_rate * ANALYSIS_WINDOW_SECONDS
    audio = np.zeros(window_samples * 2, dtype=np.float32)

    results = analyzer.analyze(audio, sample_rate)

    assert len(results) == 2
    assert_all_scores_in_range(results)


def test_analyze_long_audio_returns_one_result_per_complete_window(
    analysis_model, license_key
):
    analyzer = make_file_analyzer_or_skip(analysis_model, license_key)
    sample_rate = analysis_model.get_optimal_sample_rate()
    step_samples = analysis_model.get_optimal_num_frames(sample_rate)
    window_samples = sample_rate * ANALYSIS_WINDOW_SECONDS
    audio = np.zeros(window_samples + 2 * step_samples, dtype=np.float32)

    results = analyzer.analyze(audio, sample_rate, step_samples)

    assert len(results) == 3
    assert_all_scores_in_range(results)


def test_analyze_ignores_partial_followup_window(analysis_model, license_key):
    analyzer = make_file_analyzer_or_skip(analysis_model, license_key)
    sample_rate = analysis_model.get_optimal_sample_rate()
    step_samples = analysis_model.get_optimal_num_frames(sample_rate)
    window_samples = sample_rate * ANALYSIS_WINDOW_SECONDS
    audio = np.zeros(window_samples + step_samples - 1, dtype=np.float32)

    results = analyzer.analyze(audio, sample_rate, step_samples)

    assert len(results) == 1
    assert_all_scores_in_range(results)
