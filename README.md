# A3EM Package
```python
import a3em
```

# a3em.utils
**preprocess**(audio, sample_rate, normalization=0.7)

This function takes in an audio clip as a `numpy.ndarray`, applies a high pass filter and a low pass filter, and normalizes the signal.

```python
import librosa
from a3em.utils import preprocess

audio_path = 'test.wav'
audio, sample_rate = librosa.load(audio_path)

pre_processed_audio = preprocess(audio, sample_rate)
```

**extract_features**(audio, sample_rate)

This function takes in an audio clip as a `numpy.ndarray` and returns a dictionary containing the peak frequency in the signal, the centroid of the signal, the bandwidth of the signal, the 5% and 95% frequencies of the signal, and the Mel-Frequency Cepstral Coefficients of the signal

```python
import librosa
from a3em.utils import preprocess, extract_features

audio_path = 'test.wav'
audio, sample_rate = librosa.load(audio_path)

pre_processed_audio = preprocess(audio, sample_rate)
features = extract_features(pre_processed_audio, sample_rate)
```

# a3em.datasets

The `a3em.datasets` module provides convenient access to publicly available A3EM datasets. Datasets automatically download, cache, preprocess, and extract features on first use.

## Arden

The Arden dataset contains collar-borne AudioMoth recordings collected in June 2025 in Samburu National Reserve, Kenya.

```python
from a3em.datasets import Arden
import os

dataset = Arden(
    path=os.getenv("DATA_PATH"),
    token=os.getenv("API_TOKEN")
)
```

The first time the dataset is accessed it will automatically:

1. Download the dataset from Dryad (if necessary).
2. Extract the downloaded archives.
3. Generate metadata.
4. Extract audio clips.
5. Preprocess the clips.
6. Compute acoustic features.

Subsequent calls reuse the cached data unless `reload=True` is specified.

---

## load_data

```python
load_data(
    test_split=0.2,
    random_state=None,
    sample_rate=2000,
    rumble_only=False,
    reload=False,
    shuffle=False
)
```

Returns train/test splits of extracted acoustic features together with their labels.

```python
from a3em.datasets import Arden
import os

dataset = Arden(
    path=os.getenv("DATA_PATH"),
    token=os.getenv("API_TOKEN")
)

x_train, x_test, y_train, y_test = dataset.load_data(
    test_split=0.2,
    random_state=123,
    shuffle=True
)
```

### Parameters

| Parameter | Description |
|------------|-------------|
| `test_split` | Fraction of samples reserved for testing. |
| `random_state` | Seed used for dataset shuffling and background noise generation. |
| `sample_rate` | Sample rate used when loading audio. |
| `rumble_only` | If `True`, only elephant rumble clips are included. Otherwise background-noise clips are also generated. |
| `reload` | Forces regeneration of cached clips and features. |
| `shuffle` | Whether to shuffle samples before creating the train/test split. |

### Returns

```
x_train : pandas.DataFrame
x_test  : pandas.DataFrame
y_train : pandas.Series
y_test  : pandas.Series
```

The labels are binary:

- `0` — Background noise
- `1` — Elephant rumble

---

## load_clips

```python
load_clips(
    random_state=None,
    sample_rate=2000,
    rumble_only=False,
    reload=False
)
```

Returns all extracted audio clips together with their computed feature vectors.

```python
clips, features = dataset.load_clips(
    rumble_only=False
)
```

### Returns

- `clips` — list of NumPy arrays containing audio clips.
- `features` — `pandas.DataFrame` containing one row of extracted acoustic features per clip.

---

## Iteration

An `Arden` dataset is iterable.

```python
dataset = Arden(path, token)

dataset.__iter__()

for clip, features in dataset:
    print(len(clip))
    print(features)
```

Each iteration returns

```python
(
    numpy.ndarray,     # audio clip
    dict               # extracted acoustic features
)
```

---

## Indexing

Individual clips can be accessed by index.

```python
clip, features = dataset[10]
```

---

## Dataset Length

The total number of clips can be obtained using `len()`.

```python
len(dataset)
```

---

## Notes

- Audio clips are automatically preprocessed before feature extraction.
- Background-noise clips are randomly generated from regions that do not overlap annotated elephant calls.
- Feature extraction is performed only once unless `reload=True`.
- The dataset internally caches both audio clips and extracted features to avoid repeated computation.