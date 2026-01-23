"""Python bindings for ai-coustics SDK"""

from enum import IntEnum

import numpy as np
import numpy.typing as npt

# Custom exception types for AIC SDK errors

class ParameterOutOfRangeError(Exception):
    """Parameter value is outside the acceptable range. Check documentation for valid values."""

    message: str

class ModelNotInitializedError(Exception):
    """Model must be initialized before calling this operation. Call `Processor.initialize` first."""

    message: str

class AudioConfigUnsupportedError(Exception):
    """Audio configuration (samplerate, num_channels, num_frames) is not supported by the model."""

    message: str

class AudioConfigMismatchError(Exception):
    """Audio buffer configuration differs from the one provided during initialization."""

    message: str

class EnhancementNotAllowedError(Exception):
    """SDK key was not authorized or process failed to report usage. Check if you have internet connection."""

    message: str

class InternalError(Exception):
    """Internal error occurred. Contact support."""

    message: str

class ParameterFixedError(Exception):
    """The requested parameter is read-only for this model type and cannot be modified."""

    message: str

class LicenseFormatInvalidError(Exception):
    """License key format is invalid or corrupted. Verify the key was copied correctly."""

    message: str

class LicenseVersionUnsupportedError(Exception):
    """License version is not compatible with the SDK version. Update SDK or contact support."""

    message: str

class LicenseExpiredError(Exception):
    """License key has expired. Renew your license to continue."""

    message: str

class ModelInvalidError(Exception):
    """The model file is invalid or corrupted. Verify the file is correct."""

    message: str

class ModelVersionUnsupportedError(Exception):
    """The model file version is not compatible with this SDK version."""

    message: str

class ModelFilePathInvalidError(Exception):
    """The path to the model file is invalid."""

    message: str

class FileSystemError(Exception):
    """The model file cannot be opened due to a filesystem error. Verify that the file exists."""

    message: str

class ModelDataUnalignedError(Exception):
    """The model data is not aligned to 64 bytes."""

    message: str

class ModelDownloadError(Exception):
    """Model download error occurred."""

    message: str
    details: str

class UnknownError(Exception):
    """Unknown error code encountered."""

    message: str
    error_code: int

def get_sdk_version() -> str:
    """Returns the version of the ai-coustics core SDK library used by this package.

    Note:
        This is not necessarily the same as this package's version.

    Returns:
        The library version as a string.

    Example:
        >>> version = aic.get_sdk_version()
        >>> print(f"ai-coustics SDK version: {version}")
    """
    ...

def get_compatible_model_version() -> int:
    """Returns the model version number compatible with this SDK build.

    Returns:
        The compatible model version number.
    """
    ...

class ProcessorConfig:
    """Audio processing configuration passed to Processor.initialize().

    Use ProcessorConfig.optimal() as a starting point, then adjust fields
    to match your stream layout.
    """

    sample_rate: int
    """Sample rate in Hz (8000 - 192000)"""

    num_channels: int
    """Number of audio channels in the stream (1 for mono, 2 for stereo, etc)"""

    num_frames: int
    """Samples per channel provided to each processing call.
    Note that using a non-optimal number of frames increases latency."""

    allow_variable_frames: bool
    """Allows frame counts below num_frames at the cost of added latency"""

    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        num_frames: int,
        allow_variable_frames: bool = False,
    ) -> None:
        """Create a new ProcessorConfig instance.

        Args:
            sample_rate: Sample rate in Hz (8000 - 192000)
            num_channels: Number of audio channels
            num_frames: Samples per channel provided to each processing call
            allow_variable_frames: Allow variable frame sizes (default: False)
        """
        ...

    def __repr__(self) -> str: ...
    @staticmethod
    def optimal(
        model: Model,
        sample_rate: int | None = None,
        num_channels: int = 1,
        num_frames: int | None = None,
        allow_variable_frames: bool = False,
    ) -> ProcessorConfig:
        """Returns a ProcessorConfig pre-filled with the model's optimal settings.

        This method provides a convenient way to create a config with optimal defaults
        while allowing you to override specific parameters as needed.

        Args:
            model: The Model instance to get optimal config for
            sample_rate: Custom sample rate in Hz. If None, uses the model's optimal sample rate (default: None)
            num_channels: Number of audio channels (default: 1)
            num_frames: Custom number of frames per processing call. If None, uses the optimal frame count
                for the sample rate (default: None). Note that using non-optimal frame counts increases latency.
            allow_variable_frames: Allow variable frame sizes (default: False)

        Returns:
            ProcessorConfig with optimal settings for the given model.

        Example:
            >>> # Use all optimal defaults with stereo
            >>> config = ProcessorConfig.optimal(model, num_channels=2)
            >>> # Use custom sample rate (optimal frames calculated automatically)
            >>> config = ProcessorConfig.optimal(model, sample_rate=44100, num_channels=2)
            >>> # Use custom sample rate and frames (increases latency)
            >>> config = ProcessorConfig.optimal(model, sample_rate=48000, num_frames=512, num_channels=2)
        """
        ...

class ProcessorParameter(IntEnum):
    """Configurable parameters for audio enhancement."""

    Bypass = ...
    """Controls whether audio processing is bypassed while preserving algorithmic delay.

    When enabled, the input audio passes through unmodified, but the output is still
    delayed by the same amount as during normal processing. This ensures seamless
    transitions when toggling enhancement on/off without audible clicks or timing shifts.

    Range: 0.0 to 1.0
        - 0.0: Enhancement active (normal processing)
        - 1.0: Bypass enabled (latency-compensated passthrough)

    Default: 0.0
    """

    EnhancementLevel = ...
    """Controls the intensity of speech enhancement processing.

    Range: 0.0 to 1.0
        - 0.0: Bypass mode - original signal passes through unchanged
        - 1.0: Full enhancement - maximum noise reduction but also more audible artifacts

    Default: 1.0
    """

    VoiceGain = ...
    """Compensates for perceived volume reduction after noise removal.

    Range: 0.1 to 4.0 (linear amplitude multiplier)
        - 0.1: Significant volume reduction (-20 dB)
        - 1.0: No gain change (0 dB, default)
        - 2.0: Double amplitude (+6 dB)
        - 4.0: Maximum boost (+12 dB)

    Formula: Gain (dB) = 20 × log₁₀(value)

    Default: 1.0
    """

class VadParameter(IntEnum):
    """Configurable parameters for Voice Activity Detection."""

    SpeechHoldDuration = ...
    """Controls for how long the VAD continues to detect speech after the audio signal
    no longer contains speech.

    The VAD reports speech detected if the audio signal contained speech in at least 50%
    of the frames processed in the last speech_hold_duration seconds.

    This affects the stability of speech detected -> not detected transitions.

    Note:
        The VAD returns a value per processed buffer, so this duration is rounded
        to the closest model window length. For example, if the model has a processing window
        length of 10 ms, the VAD will round up/down to the closest multiple of 10 ms.
        Because of this, this parameter may return a different value than the one it was last set to.

    Range: 0.0 to 100x model window length (value in seconds)

    Default: 0.05 (50 ms)
    """

    Sensitivity = ...
    """Controls the sensitivity (energy threshold) of the VAD.

    This value is used by the VAD as the threshold a speech audio signal's energy
    has to exceed in order to be considered speech.

    Range: 1.0 to 15.0

    Formula: Energy threshold = 10 ^ (-sensitivity)

    Default: 6.0
    """

    MinimumSpeechDuration = ...
    """Controls for how long speech needs to be present in the audio signal before
    the VAD considers it speech.

    This affects the stability of speech not detected -> detected transitions.

    Note:
        The VAD returns a value per processed buffer, so this duration is rounded
        to the closest model window length. For example, if the model has a processing window
        length of 10 ms, the VAD will round up/down to the closest multiple of 10 ms.
        Because of this, this parameter may return a different value than the one it was last set to.

    Range: 0.0 to 1.0 (value in seconds)

    Default: 0.0
    """

class Model:
    """High-level wrapper for the ai-coustics audio enhancement model.

    This class provides a safe, Python-friendly interface to the underlying C library.
    It handles memory management automatically.

    Example:
        >>> model = Model.from_file("/path/to/model.aicmodel")
        >>> processor = Processor(model, license_key)
        >>> config = ProcessorConfig.optimal(model, num_channels=2)
        >>> processor.initialize(config)
    """

    @staticmethod
    def from_file(path: str) -> Model:
        """Creates a new audio enhancement model instance from a file.

        Multiple models can be created to process different audio streams simultaneously
        or to switch between different enhancement algorithms during runtime.

        Args:
            path: Path to the model file (.aicmodel). You can download models manually
                from https://artifacts.ai-coustics.io or use Model.download() to fetch
                them programmatically.

        Returns:
            A new Model instance.

        Raises:
            RuntimeError: If model creation fails.

        See Also:
            https://artifacts.ai-coustics.io for available model IDs and downloads.

        Example:
            >>> model = Model.from_file("/path/to/model.aicmodel")
        """
        ...

    @staticmethod
    def download(model_id: str, download_dir: str) -> str:
        """Downloads a model file from the ai-coustics artifact CDN.

        This method fetches the model manifest, verifies that the requested model
        exists in a version compatible with this library, and downloads the model
        file to the specified directory. If the model file already exists, it will not
        be re-downloaded. If the existing file's checksum does not match, the model will
        be downloaded and the existing file will be replaced.

        The manifest file is not cached and will always be downloaded on every call
        to ensure the latest model versions are always used.

        Available models can be browsed at [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io/).

        Note:
            This is a blocking operation that performs network I/O.

        Args:
            model_id: The model identifier (e.g., `"quail-l-16khz"`).
            download_dir: Directory where the model file will be stored.

        Returns:
            The full path to the model file.

        Raises:
            RuntimeError: If the operation fails.

        Example:
            >>> # Find model IDs at https://artifacts.ai-coustics.io
            >>> path = Model.download("sparrow-l-16khz", "/tmp/models")
            >>> model = Model.from_file(path)
        """
        ...

    @staticmethod
    async def download_async(model_id: str, download_dir: str) -> str:
        """Downloads a model file asynchronously from the ai-coustics artifact CDN.

        This method fetches the model manifest, verifies that the requested model
        exists in a version compatible with this library, and downloads the model
        file to the specified directory. If the model file already exists, it will not
        be re-downloaded. If the existing file's checksum does not match, the model will
        be downloaded and the existing file will be replaced.

        The manifest file is not cached and will always be downloaded on every call
        to ensure the latest model versions are always used.

        Available models can be browsed at [artifacts.ai-coustics.io](https://artifacts.ai-coustics.io/).

        Note:
            This is a blocking operation that performs network I/O.

        Args:
            model_id: The model identifier (e.g., `"quail-l-16khz"`).
            download_dir: Directory where the model file will be stored.

        Returns:
            The full path to the model file.

        Raises:
            RuntimeError: If the operation fails.

        Example:
            >>> # Find model IDs at https://artifacts.ai-coustics.io
            >>> path = await Model.download_async("sparrow-l-16khz", "/tmp/models")
            >>> model = Model.from_file(path)
        """
        ...

    def get_id(self) -> str:
        """Returns the model identifier string.

        Returns:
            The model ID string.
        """
        ...

    def get_optimal_sample_rate(self) -> int:
        """Retrieves the native sample rate of the model.

        Each model is optimized for a specific sample rate, which determines the frequency
        range of the enhanced audio output. While you can process audio at any sample rate,
        understanding the model's native rate helps predict the enhancement quality.

        How sample rate affects enhancement:
            - Models trained at lower sample rates (e.g., 8 kHz) can only enhance frequencies
              up to their Nyquist limit (4 kHz for 8 kHz models)
            - When processing higher sample rate input (e.g., 48 kHz) with a lower-rate model,
              only the lower frequency components will be enhanced

        Enhancement blending:
            When enhancement strength is set below 1.0, the enhanced signal is blended with
            the original, maintaining the full frequency spectrum of your input while adding
            the model's noise reduction capabilities to the lower frequencies.

        Sample rate and optimal frames relationship:
            When using different sample rates than the model's native rate, the optimal number
            of frames (returned by get_optimal_num_frames) will change. The model's output
            delay remains constant regardless of sample rate as long as you use the optimal frame
            count for that rate.

        Recommendation:
            For maximum enhancement quality across the full frequency spectrum, match your
            input sample rate to the model's native rate when possible.

        Returns:
            The model's native sample rate in Hz.

        Example:
            >>> optimal_rate = model.get_optimal_sample_rate()
            >>> print(f"Optimal sample rate: {optimal_rate} Hz")
        """
        ...

    def get_optimal_num_frames(self, sample_rate: int) -> int:
        """Retrieves the optimal number of frames for the model at a given sample rate.

        Using the optimal number of frames minimizes latency by avoiding internal buffering.

        When you use a different frame count than the optimal value, the model will
        introduce additional buffering latency on top of its base processing delay.

        The optimal frame count varies based on the sample rate. Each model operates on a
        fixed time window duration, so the required number of frames changes with sample rate.
        For example, a model designed for 10 ms processing windows requires 480 frames at
        48 kHz, but only 160 frames at 16 kHz to capture the same duration of audio.

        Call this function with your intended sample rate before calling
        Processor.initialize() to determine the best frame count for minimal latency.

        Args:
            sample_rate: The sample rate in Hz for which to calculate the optimal frame count

        Returns:
            The optimal frame count for the given sample rate.

        Example:
            >>> sample_rate = model.get_optimal_sample_rate()
            >>> optimal_frames = model.get_optimal_num_frames(sample_rate)
            >>> print(f"Optimal frame count: {optimal_frames}")
        """
        ...

class ProcessorContext:
    """Context for managing processor state and parameters.

    Created via Processor.create_processor_context().
    """

    def reset(self) -> None:
        """Clears all internal state and buffers.

        Call this when the audio stream is interrupted or when seeking
        to prevent artifacts from previous audio content.

        The processor stays initialized to the configured settings.

        Thread Safety:
            Real-time safe. Can be called from audio processing threads.

        Example:
            >>> processor_context.reset()
        """
        ...

    def set_parameter(self, parameter: ProcessorParameter, value: float) -> None:
        """Modifies a processor parameter.

        All parameters can be changed during audio processing.
        This function can be called from any thread.

        Args:
            parameter: Parameter to modify
            value: New parameter value. See parameter documentation for ranges

        Raises:
            ValueError: If the parameter value is out of range.

        Example:
            >>> processor_context.set_parameter(ProcessorParameter.EnhancementLevel, 0.8)
        """
        ...

    def get_parameter(self, parameter: ProcessorParameter) -> float:
        """Retrieves the current value of a parameter.

        This function can be called from any thread.

        Args:
            parameter: Parameter to query

        Returns:
            The current parameter value.

        Example:
            >>> level = processor_context.get_parameter(ProcessorParameter.EnhancementLevel)
            >>> print(f"Current enhancement level: {level}")
        """
        ...

    def get_output_delay(self) -> int:
        """Returns the total output delay in samples for the current audio configuration.

        This function provides the complete end-to-end latency introduced by the model,
        which includes both algorithmic processing delay and any buffering overhead.
        Use this value to synchronize enhanced audio with other streams or to implement
        delay compensation in your application.

        Delay behavior:
            - Before initialization: Returns the base processing delay using the model's
              optimal frame size at its native sample rate
            - After initialization: Returns the actual delay for your specific configuration,
              including any additional buffering introduced by non-optimal frame sizes

        Important:
            The delay value is always expressed in samples at the sample rate
            you configured during initialize(). To convert to time units:
            delay_ms = (delay_samples * 1000) / sample_rate

        Note:
            Using frame sizes different from the optimal value returned by
            get_optimal_num_frames() will increase the delay beyond the model's base latency.

        Returns:
            The delay in samples.

        Example:
            >>> delay = processor_context.get_output_delay()
            >>> print(f"Output delay: {delay} samples")
        """
        ...

class VadContext:
    """Voice Activity Detector backed by an ai-coustics speech enhancement model.

    The VAD works automatically using the enhanced audio output of the model
    that created the VAD.

    Important:
        - The latency of the VAD prediction is equal to the backing model's processing latency.
        - If the backing model stops being processed, the VAD will not update its speech detection prediction.

    Created via Processor.create_vad_context().

    Example:
        >>> vad = processor.create_vad_context()
        >>> vad.set_parameter(VadParameter.Sensitivity, 5.0)
        >>> if vad.is_speech_detected():
        ...     print("Speech detected!")
    """

    def is_speech_detected(self) -> bool:
        """Returns the VAD's prediction.

        Important:
            - The latency of the VAD prediction is equal to the backing model's processing latency.
            - If the backing model stops being processed, the VAD will not update its speech detection prediction.

        Returns:
            True if speech is detected, False otherwise.
        """
        ...

    def set_parameter(self, parameter: VadParameter, value: float) -> None:
        """Modifies a VAD parameter.

        Args:
            parameter: Parameter to modify
            value: New parameter value. See parameter documentation for ranges

        Raises:
            ValueError: If the parameter value is out of range.

        Example:
            >>> vad.set_parameter(VadParameter.SpeechHoldDuration, 0.08)
            >>> vad.set_parameter(VadParameter.Sensitivity, 5.0)
        """
        ...

    def get_parameter(self, parameter: VadParameter) -> float:
        """Retrieves the current value of a VAD parameter.

        Args:
            parameter: Parameter to query

        Returns:
            The current parameter value.

        Example:
            >>> sensitivity = vad.get_parameter(VadParameter.Sensitivity)
            >>> print(f"Current sensitivity: {sensitivity}")
        """
        ...

class Processor:
    """High-level wrapper for the ai-coustics audio enhancement processor.

    This class provides a safe, Python-friendly interface to the underlying C library.
    It handles memory management automatically.

    Example:
        >>> model = Model.from_file("/path/to/model.aicmodel")
        >>> processor = Processor(model, license_key)
        >>> config = ProcessorConfig.optimal(model, num_channels=2)
        >>> processor.initialize(config)
        >>> audio = np.zeros((2, config.num_frames), dtype=np.float32)
        >>> enhanced = processor.process(audio)
    """

    def __init__(
        self, model: Model, license_key: str, config: ProcessorConfig | None = None
    ) -> None:
        """Creates a new audio enhancement processor instance.

        Multiple processors can be created to process different audio streams simultaneously
        or to switch between different enhancement algorithms during runtime.

        If a config is provided, the processor will be initialized immediately.
        Otherwise, you must call initialize() before processing audio.

        Args:
            model: The loaded model instance
            license_key: License key for the ai-coustics SDK
                (generate your key at https://developers.ai-coustics.com/)
            config: Optional audio processing configuration. If provided, the processor
                will be initialized immediately with this configuration.

        Raises:
            RuntimeError: If processor creation fails.
            ValueError: If config is provided and the audio configuration is unsupported.

        Example:
            >>> # Create processor without initialization
            >>> processor = Processor(model, license_key)
            >>> processor.initialize(config)

            >>> # Or create and initialize in one step
            >>> config = ProcessorConfig.optimal(model, num_channels=2)
            >>> processor = Processor(model, license_key, config)
        """
        ...

    def initialize(self, config: ProcessorConfig) -> None:
        """Configures the processor for specific audio settings.

        This function must be called before processing any audio.
        For the lowest delay use the sample rate and frame size returned by
        Model.get_optimal_sample_rate() and Model.get_optimal_num_frames().

        Args:
            config: Audio processing configuration

        Raises:
            ValueError: If the audio configuration is unsupported.

        Warning:
            Do not call from audio processing threads as this allocates memory.

        Note:
            All channels are mixed to mono for processing. To process channels
            independently, create separate Processor instances.

        Example:
            >>> config = ProcessorConfig.optimal(model)
            >>> processor.initialize(config)
        """
        ...

    def process(self, buffer: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Processes audio from a 2D NumPy array (channels × frames).

        Enhances speech in the provided audio buffer and returns a new array
        with the processed audio data.

        The input uses sequential channel layout where all samples for each
        channel are stored contiguously.

        Args:
            buffer: 2D NumPy array with shape (num_channels, num_frames) containing
                   audio data to be enhanced

        Returns:
            A new NumPy array with the same shape containing the enhanced audio.

        Raises:
            ModelNotInitializedError: If the processor has not been initialized.
            AudioConfigMismatchError: If the buffer shape doesn't match the configured audio settings.
            EnhancementNotAllowedError: If SDK key is not authorized or processing fails to report usage.
            InternalError: If an internal processing error occurs.

        Example:
            >>> audio = np.random.randn(2, 1024).astype(np.float32)
            >>> enhanced = processor.process(audio)
        """
        ...

    def get_processor_context(self) -> ProcessorContext:
        """Creates a ProcessorContext instance.

        This can be used to control all parameters and other settings of the processor.

        Returns:
            A new ProcessorContext instance.

        Example:
            >>> processor_context = processor.get_processor_context()
        """
        ...

    def get_vad_context(self) -> VadContext:
        """Creates a Voice Activity Detector Context instance.

        Returns:
            A new VadContext instance.

        Example:
            >>> vad = processor.get_vad_context()
        """
        ...

class ProcessorAsync:
    """Async wrapper for Processor that offloads processing to a thread pool.

    This class provides the same functionality as Processor but with async methods
    that don't block the event loop.

    Example:
        >>> model = Model.from_file("/path/to/model.aicmodel")
        >>> processor = ProcessorAsync(model, license_key)
        >>> config = ProcessorConfig.optimal(model, num_channels=2)
        >>> await processor.initialize_async(config)
        >>> audio = np.zeros((2, config.num_frames), dtype=np.float32)
        >>> enhanced = await processor.process_async(audio)
    """

    def __init__(
        self, model: Model, license_key: str, config: ProcessorConfig | None = None
    ) -> None:
        """Creates a new async audio enhancement processor instance.

        Multiple processors can be created to process different audio streams simultaneously
        or to switch between different enhancement algorithms during runtime.

        If a config is provided, the processor will be initialized immediately.
        Otherwise, you must call initialize_async() before processing audio.

        Args:
            model: The loaded model instance
            license_key: License key for the ai-coustics SDK
                (generate your key at https://developers.ai-coustics.io/)
            config: Optional audio processing configuration. If provided, the processor
                will be initialized immediately with this configuration.

        Raises:
            RuntimeError: If processor creation fails.
            ValueError: If config is provided and the audio configuration is unsupported.

        Example:
            >>> # Create processor without initialization
            >>> processor = ProcessorAsync(model, license_key)
            >>> await processor.initialize_async(config)

            >>> # Or create and initialize in one step
            >>> config = ProcessorConfig.optimal(model, num_channels=2)
            >>> processor = ProcessorAsync(model, license_key, config)
        """
        ...

    async def initialize_async(self, config: ProcessorConfig) -> None:
        """Configures the processor asynchronously for specific audio settings.

        This function must be called before processing any audio.
        For the lowest delay use the sample rate and frame size returned by
        Model.get_optimal_sample_rate() and Model.get_optimal_num_frames().

        Args:
            config: Audio processing configuration

        Raises:
            ValueError: If the audio configuration is unsupported.

        Note:
            All channels are mixed to mono for processing. To process channels
            independently, create separate ProcessorAsync instances.

        Example:
            >>> config = ProcessorConfig.optimal(model)
            >>> await processor.initialize_async(config)
        """
        ...

    async def process_async(
        self,
        buffer: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Processes audio asynchronously from a 2D NumPy array (channels × frames).

        Enhances speech in the provided audio buffer and returns a new array
        with the processed audio data. Processing happens in a background thread.

        The input uses sequential channel layout where all samples for each
        channel are stored contiguously.

        Args:
            buffer: 2D NumPy array with shape (num_channels, num_frames) containing
                   audio data to be enhanced

        Returns:
            A new NumPy array with the same shape containing the enhanced audio.

        Raises:
            ModelNotInitializedError: If the processor has not been initialized.
            AudioConfigMismatchError: If the buffer shape doesn't match the configured audio settings.
            EnhancementNotAllowedError: If SDK key is not authorized or processing fails to report usage.
            InternalError: If an internal processing error occurs.

        Example:
            >>> audio = np.random.randn(2, 1024).astype(np.float32)
            >>> enhanced = await processor.process_async(audio)
        """
        ...

    def get_processor_context(self) -> ProcessorContext:
        """Creates a ProcessorContext instance.

        This can be used to control all parameters and other settings of the processor.

        Returns:
            A new ProcessorContext instance.

        Example:
            >>> processor_context = processor.get_processor_context()
        """
        ...

    def get_vad_context(self) -> VadContext:
        """Creates a Voice Activity Detector Context instance.

        Returns:
            A new VadContext instance.

        Example:
            >>> vad = processor.get_vad_context()
        """
        ...
