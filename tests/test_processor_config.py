import aic_sdk as aic


def test_processor_config_init_sets_sample_rate():
    config = aic.ProcessorConfig(48000, 480, False)
    assert config.sample_rate == 48000


def test_processor_config_init_sets_block_size():
    config = aic.ProcessorConfig(48000, 480, False)
    assert config.block_size == 480


def test_processor_config_init_sets_variable_block_size():
    config = aic.ProcessorConfig(48000, 480, True)
    assert config.variable_block_size is True


def test_processor_config_variable_block_size_defaults_to_false():
    config = aic.ProcessorConfig(48000, 480)
    assert config.variable_block_size is False


def test_processor_config_sample_rate_is_mutable():
    config = aic.ProcessorConfig(48000, 480, False)
    config.sample_rate = 16000
    assert config.sample_rate == 16000


def test_processor_config_block_size_is_mutable():
    config = aic.ProcessorConfig(48000, 480, False)
    config.block_size = 960
    assert config.block_size == 960


def test_processor_config_variable_block_size_is_mutable():
    config = aic.ProcessorConfig(48000, 480, False)
    config.variable_block_size = True
    assert config.variable_block_size is True


def test_processor_config_repr_contains_sample_rate():
    config = aic.ProcessorConfig(48000, 480, False)
    assert "48000" in repr(config)


def test_processor_config_repr_contains_block_size():
    config = aic.ProcessorConfig(48000, 480, False)
    assert "480" in repr(config)


def test_processor_config_optimal_returns_config(model):
    config = aic.ProcessorConfig.optimal(model)
    assert isinstance(config, aic.ProcessorConfig)


def test_processor_config_optimal_uses_model_sample_rate(model):
    config = aic.ProcessorConfig.optimal(model)
    assert config.sample_rate == model.get_optimal_sample_rate()


def test_processor_config_optimal_uses_model_block_size(model):
    config = aic.ProcessorConfig.optimal(model)
    expected_block_size = model.get_optimal_block_size(config.sample_rate)
    assert config.block_size == expected_block_size


def test_processor_config_optimal_defaults_variable_block_size_false(model):
    config = aic.ProcessorConfig.optimal(model)
    assert config.variable_block_size is False


def test_processor_config_optimal_accepts_variable_block_size(model):
    config = aic.ProcessorConfig.optimal(model, variable_block_size=True)
    assert config.variable_block_size is True
