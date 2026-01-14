import aic_sdk as aic


def test_processor_config_init_sets_sample_rate():
    config = aic.ProcessorConfig(48000, 2, 480, False)
    assert config.sample_rate == 48000


def test_processor_config_init_sets_num_channels():
    config = aic.ProcessorConfig(48000, 2, 480, False)
    assert config.num_channels == 2


def test_processor_config_init_sets_num_frames():
    config = aic.ProcessorConfig(48000, 2, 480, False)
    assert config.num_frames == 480


def test_processor_config_init_sets_allow_variable_frames():
    config = aic.ProcessorConfig(48000, 2, 480, True)
    assert config.allow_variable_frames is True


def test_processor_config_allow_variable_frames_defaults_to_false():
    config = aic.ProcessorConfig(48000, 1, 480)
    assert config.allow_variable_frames is False


def test_processor_config_sample_rate_is_mutable():
    config = aic.ProcessorConfig(48000, 1, 480, False)
    config.sample_rate = 16000
    assert config.sample_rate == 16000


def test_processor_config_num_channels_is_mutable():
    config = aic.ProcessorConfig(48000, 1, 480, False)
    config.num_channels = 2
    assert config.num_channels == 2


def test_processor_config_num_frames_is_mutable():
    config = aic.ProcessorConfig(48000, 1, 480, False)
    config.num_frames = 960
    assert config.num_frames == 960


def test_processor_config_allow_variable_frames_is_mutable():
    config = aic.ProcessorConfig(48000, 1, 480, False)
    config.allow_variable_frames = True
    assert config.allow_variable_frames is True


def test_processor_config_repr_contains_sample_rate():
    config = aic.ProcessorConfig(48000, 2, 480, False)
    assert "48000" in repr(config)


def test_processor_config_repr_contains_num_channels():
    config = aic.ProcessorConfig(48000, 2, 480, False)
    assert "2" in repr(config)


def test_processor_config_repr_contains_num_frames():
    config = aic.ProcessorConfig(48000, 2, 480, False)
    assert "480" in repr(config)


def test_processor_config_optimal_returns_config(model):
    config = aic.ProcessorConfig.optimal(model)
    assert isinstance(config, aic.ProcessorConfig)


def test_processor_config_optimal_uses_model_sample_rate(model):
    config = aic.ProcessorConfig.optimal(model)
    assert config.sample_rate == model.get_optimal_sample_rate()


def test_processor_config_optimal_uses_model_num_frames(model):
    config = aic.ProcessorConfig.optimal(model)
    expected_frames = model.get_optimal_num_frames(config.sample_rate)
    assert config.num_frames == expected_frames


def test_processor_config_optimal_defaults_to_mono(model):
    config = aic.ProcessorConfig.optimal(model)
    assert config.num_channels == 1


def test_processor_config_optimal_accepts_num_channels(model):
    config = aic.ProcessorConfig.optimal(model, num_channels=2)
    assert config.num_channels == 2


def test_processor_config_optimal_defaults_allow_variable_frames_false(model):
    config = aic.ProcessorConfig.optimal(model)
    assert config.allow_variable_frames is False


def test_processor_config_optimal_accepts_allow_variable_frames(model):
    config = aic.ProcessorConfig.optimal(model, allow_variable_frames=True)
    assert config.allow_variable_frames is True
