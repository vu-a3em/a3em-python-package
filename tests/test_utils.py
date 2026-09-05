import numpy as np
import pytest

from a3em_analysis.utils import preprocess


@pytest.fixture
def sample_audio():
    sample_rate = 8000
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    return np.sin(2 * np.pi * 40 * t), sample_rate


def test_preprocess_returns_same_shape(sample_audio):
    audio, sample_rate = sample_audio
    result = preprocess(audio, sample_rate)
    assert result.shape == audio.shape


def test_preprocess_normalizes_peak_amplitude(sample_audio):
    audio, sample_rate = sample_audio
    normalization = 0.5
    result = preprocess(audio, sample_rate, normalization=normalization)
    assert np.max(np.abs(result)) == pytest.approx(normalization)


@pytest.mark.parametrize("normalization", [-0.1, 1.1])
def test_preprocess_rejects_invalid_normalization(sample_audio, normalization):
    audio, sample_rate = sample_audio
    with pytest.raises(ValueError):
        preprocess(audio, sample_rate, normalization=normalization)
