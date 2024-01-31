import abc
import io
import math
import os
from typing import Union
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio

from asr.utils import env_util, math_util
from asr.featurizers import gammatone


# def tf_resample(signal, rate_in, rate_out):
#     if rate_in == rate_out: return signal
#     rate_in = tf.cast(rate_in, dtype=tf.float32)
#     rate_out = tf.cast(rate_out, dtype=tf.float32)
#     ratio = rate_out / rate_in
#     nsamples = tf.math.ceil(tf.shape(signal)[0] * ratio)


def load_and_convert_to_wav(
    path: str,
) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)


def read_raw_audio(
    audio: Union[str, bytes, np.ndarray],
    sample_rate=16000,
) -> np.ndarray:
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1:
            wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1:
            ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def tf_read_raw_audio(
    audio: tf.Tensor,
    sample_rate=16000,
) -> tf.Tensor:
    wave, rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    if not env_util.has_devices("TPU"):
        resampled = tfio.audio.resample(wave, rate_in=tf.cast(rate, dtype=tf.int64), rate_out=sample_rate)
        return tf.reshape(resampled, shape=[-1])  # reshape for using tf.signal
    return tf.reshape(wave, shape=[-1])  # reshape for using tf.signal


def slice_signal(
    signal,
    window_size,
    stride=0.5,
) -> np.ndarray:
    """Return windows of the given signal by sweeping in stride fractions of window"""
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset), range(window_size, n_samples + offset, offset)):
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] < window_size:
            slice_ = np.pad(slice_, (0, window_size - slice_.shape[0]), "constant", constant_values=0.0)
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float32)


def tf_merge_slices(
    slices: tf.Tensor,
) -> tf.Tensor:
    # slices shape = [batch, window_size]
    return tf.keras.backend.flatten(slices)  # return shape = [-1, ]


def merge_slices(
    slices: np.ndarray,
) -> np.ndarray:
    # slices shape = [batch, window_size]
    return np.reshape(slices, [-1])


def normalize_audio_feature(
    audio_feature: np.ndarray,
    per_frame=False,
) -> np.ndarray:
    """Mean and variance normalization"""
    axis = 1 if per_frame else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.sqrt(np.var(audio_feature, axis=axis) + 1e-9)
    normalized = (audio_feature - mean) / std_dev
    return normalized


def tf_normalize_audio_features(
    audio_feature: tf.Tensor,
    per_frame=False,
) -> tf.Tensor:
    """
    TF Mean and variance features normalization
    Args:
        audio_feature: tf.Tensor with shape [T, F]

    Returns:
        normalized audio features with shape [T, F]
    """
    axis = 1 if per_frame else None
    mean = tf.reduce_mean(audio_feature, axis=axis, keepdims=True)
    std_dev = tf.math.sqrt(tf.math.reduce_variance(audio_feature, axis=axis, keepdims=True) + 1e-9)
    return (audio_feature - mean) / std_dev


def normalize_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """Normailize signal to [-1, 1] range"""
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


def tf_normalize_signal(
    signal: tf.Tensor,
) -> tf.Tensor:
    """
    TF Normailize signal to [-1, 1] range
    Args:
        signal: tf.Tensor with shape [None]

    Returns:
        normalized signal with shape [None]
    """
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
    return signal * gain


def preemphasis(
    signal: np.ndarray,
    coeff=0.97,
) -> np.ndarray:
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def tf_preemphasis(
    signal: tf.Tensor,
    coeff=0.97,
):
    """
    TF Pre-emphasis
    Args:
        signal: tf.Tensor with shape [None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        pre-emphasized signal with shape [None]
    """
    if not coeff or coeff <= 0.0:
        return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis=-1)


def depreemphasis(
    signal: np.ndarray,
    coeff=0.97,
) -> np.ndarray:
    if not coeff or coeff <= 0.0:
        return signal
    x = np.zeros(signal.shape[0], dtype=np.float32)
    x[0] = signal[0]
    for n in range(1, signal.shape[0], 1):
        x[n] = coeff * x[n - 1] + signal[n]
    return x


def tf_depreemphasis(
    signal: tf.Tensor,
    coeff=0.97,
) -> tf.Tensor:
    """
    TF Depreemphasis
    Args:
        signal: tf.Tensor with shape [B, None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        depre-emphasized signal with shape [B, None]
    """
    if not coeff or coeff <= 0.0:
        return signal

    def map_fn(elem):
        x = tf.expand_dims(elem[0], axis=-1)
        for n in range(1, elem.shape[0], 1):
            current = coeff * x[n - 1] + elem[n]
            x = tf.concat([x, [current]], axis=0)
        return x

    return tf.map_fn(map_fn, signal)


class SpeechFeaturizer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        speech_config: dict,
    ):
        """
        We should use TFSpeechFeaturizer for training to avoid differences
        between tf and librosa when converting to tflite in post-training stage
        speech_config = {
            "sample_rate": int,
            "frame_ms": int,
            "stride_ms": int,
            "num_feature_bins": int,
            "feature_type": str,
            "delta": bool,
            "delta_delta": bool,
            "pitch": bool,
            "normalize_signal": bool,
            "normalize_feature": bool,
            "normalize_per_frame": bool
        }
        """
        # Samples
        self.sample_rate = speech_config.get("sample_rate", 16000)
        self.frame_length = int(self.sample_rate * (speech_config.get("frame_ms", 25) / 1000))
        self.frame_step = int(self.sample_rate * (speech_config.get("stride_ms", 10) / 1000))
        # Features
        self.num_feature_bins = speech_config.get("num_feature_bins", 80)
        self.feature_type = speech_config.get("feature_type", "log_mel_spectrogram")
        self.preemphasis = speech_config.get("preemphasis", None)
        self.top_db = speech_config.get("top_db", 80.0)
        # Normalization
        self.normalize_signal = speech_config.get("normalize_signal", True)
        self.normalize_feature = speech_config.get("normalize_feature", True)
        self.normalize_per_frame = speech_config.get("normalize_per_frame", False)
        self.center = speech_config.get("center", True)
        # Length
        self.max_length = 0

    @property
    def nfft(self) -> int:
        """Number of FFT"""
        return 2 ** (self.frame_length - 1).bit_length()

    @property
    def shape(self) -> list:
        """The shape of extracted features"""
        raise NotImplementedError()

    def get_length_from_duration(self, duration):
        nsamples = math.ceil(float(duration) * self.sample_rate)
        if self.center:
            nsamples += self.nfft
        return 1 + (nsamples - self.nfft) // self.frame_step  # https://www.tensorflow.org/api_docs/python/tf/signal/frame

    def update_length(self, length: int):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    @abc.abstractclassmethod
    def stft(self, signal):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        raise NotImplementedError()

    @abc.abstractmethod
    def extract(self, signal):
        """Function to perform feature extraction"""
        raise NotImplementedError()


class TFSpeechFeaturizer(SpeechFeaturizer):
    @property
    def shape(self) -> list:
        length = self.max_length if self.max_length > 0 else None
        return [length, self.num_feature_bins, 1]

    def stft(
        self,
        signal,
    ):
        if self.center:
            signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode="REFLECT")
        window = tf.signal.hann_window(self.frame_length, periodic=True)
        left_pad = (self.nfft - self.frame_length) // 2
        right_pad = self.nfft - self.frame_length - left_pad
        window = tf.pad(window, [[left_pad, right_pad]])
        framed_signals = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.frame_step)
        framed_signals *= window
        return tf.square(tf.abs(tf.signal.rfft(framed_signals, [self.nfft])))

    def power_to_db(
        self,
        S,
        amin=1e-10,
    ):
        log_spec = 10.0 * math_util.log10(tf.maximum(amin, S))
        log_spec -= 10.0 * math_util.log10(tf.maximum(amin, 1.0))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - self.top_db)

        return log_spec

    def extract(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        signal = np.asfortranarray(signal)
        features = self.tf_extract(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()

    def tf_extract(
        self,
        signal: tf.Tensor,
    ) -> tf.Tensor:
        """
        Extract speech features from signals (for using in tflite)
        Args:
            signal: tf.Tensor with shape [None]

        Returns:
            features: tf.Tensor with shape [T, F, 1]
        """
        if self.normalize_signal:
            signal = tf_normalize_signal(signal)
        signal = tf_preemphasis(signal, self.preemphasis)

        if self.feature_type == "spectrogram":
            features = self.compute_spectrogram(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self.compute_log_mel_spectrogram(signal)
        elif self.feature_type == "mfcc":
            features = self.compute_mfcc(signal)
        elif self.feature_type == "log_gammatone_spectrogram":
            features = self.compute_log_gammatone_spectrogram(signal)
        else:
            raise ValueError("feature_type must be either 'mfcc', 'log_mel_spectrogram' or 'spectrogram'")

        features = tf.expand_dims(features, axis=-1)

        if self.normalize_feature:
            features = tf_normalize_audio_features(features, per_frame=self.normalize_per_frame)

        return features

    def compute_log_mel_spectrogram(
        self,
        signal,
    ):
        spectrogram = self.stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=(self.sample_rate / 2),
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        return self.power_to_db(mel_spectrogram)

    def compute_spectrogram(
        self,
        signal,
    ):
        S = self.stft(signal)
        spectrogram = self.power_to_db(S)
        return spectrogram[:, : self.num_feature_bins]

    def compute_mfcc(
        self,
        signal,
    ):
        log_mel_spectrogram = self.compute_log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    def compute_log_gammatone_spectrogram(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        S = self.stft(signal)

        gtone = gammatone.fft_weights(
            self.nfft,
            self.sample_rate,
            self.num_feature_bins,
            width=1.0,
            fmin=0,
            fmax=int(self.sample_rate / 2),
            maxlen=(self.nfft / 2 + 1),
        )

        gtone_spectrogram = tf.tensordot(S, gtone, 1)

        return self.power_to_db(gtone_spectrogram)
