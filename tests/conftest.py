import numpy as np
import pandas as pd
import soundfile as sf
import pytest

from a3em_analysis.datasets import Arden

SAMPLE_RATE = 2000


def _make_recording(path, duration_s, freq=40.0):
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    audio = 0.8 * np.sin(2 * np.pi * freq * t)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")


def _make_annotations(path, rumble_windows):
    rows = [
        {
            "Selection": i + 1,
            "View": "Spectrogram 1",
            "Channel": 1,
            "Begin Time (s)": begin,
            "End Time (s)": end,
            "Low Freq (Hz)": 10,
            "High Freq (Hz)": 80,
            "call_type": "RUM",
            "quality": quality,
            "overlap": "N",
            "earflap": 0,
        }
        for i, (begin, end, quality) in enumerate(rumble_windows)
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def arden_dataset_dir(tmp_path):
    """Builds a small local Arden-shaped dataset so Arden can __setup()
    without hitting the network or Data Dryad."""
    audiomoth_path = tmp_path / "audiomoth"
    annotations_path = tmp_path / "manualAnnotations"
    audiomoth_path.mkdir()
    annotations_path.mkdir()

    recordings = {
        "rec1": [(1.0, 4.0, 2), (10.0, 15.0, 3)],
        "rec2": [(2.0, 5.0, 2), (11.0, 16.0, 4)],
    }

    for stem, rumble_windows in recordings.items():
        _make_recording(audiomoth_path / f"{stem}.WAV", duration_s=20.0)
        _make_annotations(annotations_path / f"{stem}.txt", rumble_windows)

    return tmp_path


@pytest.fixture
def make_arden(arden_dataset_dir):
    """Factory for fresh, independent Arden instances pointed at the same
    on-disk dataset, so determinism can be checked across instances."""
    def _make():
        return Arden(arden_dataset_dir, token="test-token")
    return _make
