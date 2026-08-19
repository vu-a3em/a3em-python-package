# A3EM Package

```python
import a3em
```

A3EM provides utilities for preprocessing bioacoustic recordings, extracting acoustic features, and working with supported bioacoustic datasets.

# `a3em.utils`

## `preprocess`

```python
preprocess(audio, sample_rate, normalization=0.7)
```

Takes an audio clip as a `numpy.ndarray`, applies high-pass and low-pass filtering, and normalizes the signal.

### Example

```python
import librosa
from a3em.utils import preprocess

audio_path = "test.wav"
audio, sample_rate = librosa.load(audio_path)

preprocessed_audio = preprocess(audio, sample_rate)
```

---

## `extract_features`

```python
extract_features(audio, sample_rate)
```

Takes an audio clip as a `numpy.ndarray` and extracts acoustic features from the signal.

### Example

```python
import librosa
from a3em.utils import preprocess, extract_features

audio_path = "test.wav"
audio, sample_rate = librosa.load(audio_path)

preprocessed_audio = preprocess(audio, sample_rate)
features = extract_features(preprocessed_audio, sample_rate)
```

# `a3em.datasets`

The `a3em.datasets` module provides convenient access to supported A3EM datasets. Dataset classes handle downloading, preprocessing, clip extraction, and feature extraction.

## Arden

The `Arden` dataset contains collar-borne AudioMoth recordings collected in June 2025 in Samburu National Reserve, Kenya.

```python
from a3em.datasets import Arden
import os

dataset = Arden(
    path=os.getenv("DATA_PATH"),
    token=os.getenv("API_TOKEN")
)
```

`path` specifies where the dataset should be stored locally. `token` is used to authenticate downloads from Dryad.

The first time data is loaded, the dataset will automatically:

1. Check for the required audio and annotation files locally.
2. Download the dataset from Dryad if necessary.
3. Extract downloaded archives.
4. Load and filter annotation metadata.
5. Optionally generate background-noise examples.
6. Extract audio clips.
7. Preprocess the clips.
8. Compute acoustic features.

The extracted clips and features are cached on the `Arden` instance so that subsequent calls do not repeat feature extraction unless `reload=True` is specified.

---

## `load_data`

```python
load_data(
    random_state=None,
    sample_rate=2000,
    rumble_only=False,
    reload=False
)
```

Loads the dataset and returns the extracted acoustic features together with their binary labels.

### Example

```python
features, labels = dataset.load_data(
    random_state=123
)
```

### Parameters

| Parameter      | Description                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `random_state` | Seed used when shuffling metadata and generating background-noise clips.                                           |
| `sample_rate`  | Sample rate used when loading audio. Defaults to `2000`.                                                           |
| `rumble_only`  | If `True`, only annotated elephant rumbles are included. If `False`, background-noise examples are also generated. |
| `reload`       | If `True`, regenerates metadata, clips, and features even if they have already been loaded.                        |

### Returns

```text
features : pandas.DataFrame
labels   : pandas.Series
```

Labels are binary:

* `0` — Background noise
* `1` — Elephant rumble

Annotation quality values `2`, `3`, and `4` are mapped to the rumble label `1`. Background-noise examples have quality `0` and are mapped to label `0`.

---

## `load_data_ml`

```python
load_data_ml(
    test_split=0.2,
    random_state=None,
    sample_rate=2000,
    rumble_only=False,
    reload=False,
    shuffle=False
)
```

Loads the dataset and creates train/test splits suitable for machine-learning workflows.

Internally, this method uses `sklearn.model_selection.train_test_split`.

### Example

```python
x_train, x_test, y_train, y_test = dataset.load_data_ml(
    test_split=0.2,
    random_state=123,
    shuffle=True
)
```

### Parameters

| Parameter      | Description                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `test_split`   | Fraction of samples reserved for testing. Defaults to `0.2`.                                                       |
| `random_state` | Seed used for dataset generation and the train/test split.                                                         |
| `sample_rate`  | Sample rate used when loading audio. Defaults to `2000`.                                                           |
| `rumble_only`  | If `True`, only annotated elephant rumbles are included. If `False`, background-noise examples are also generated. |
| `reload`       | If `True`, regenerates metadata, clips, and features before splitting.                                             |
| `shuffle`      | Whether samples should be shuffled by `train_test_split` before creating the split. Defaults to `False`.           |

### Returns

```text
x_train : pandas.DataFrame
x_test  : pandas.DataFrame
y_train : pandas.Series
y_test  : pandas.Series
```

---

## `load_clips`

```python
load_clips(
    random_state=None,
    sample_rate=2000,
    rumble_only=False,
    reload=False
)
```

Returns the extracted audio clips together with their computed acoustic features.

### Example

```python
clips, features = dataset.load_clips(
    random_state=123,
    rumble_only=False
)
```

### Parameters

| Parameter      | Description                                              |
| -------------- | -------------------------------------------------------- |
| `random_state` | Seed used when generating the dataset.                   |
| `sample_rate`  | Sample rate used when loading audio. Defaults to `2000`. |
| `rumble_only`  | If `True`, background-noise examples are not generated.  |
| `reload`       | If `True`, regenerates the clips and features.           |

### Returns

* `clips` — list containing the extracted audio clips as NumPy arrays.
* `features` — `pandas.DataFrame` containing one row of acoustic features for each clip.

Each extracted clip includes a `0.2` second buffer before and after its annotated time range.

Clips shorter than two seconds after extraction are discarded during feature extraction.

---

## Iteration

An `Arden` dataset can be iterated over directly.

```python
dataset = Arden(path, token)

for clip, features in dataset:
    print(len(clip))
    print(features)
```

If the dataset has not already been loaded, iteration automatically initializes it using the default loading options.

Each iteration returns:

```python
(
    numpy.ndarray,  # audio clip
    dict            # extracted acoustic features
)
```

---

## Indexing

Individual clips and their corresponding features can be accessed by index after the dataset has been loaded.

```python
clip, features = dataset[10]
```

The returned feature set is converted from its DataFrame row into a dictionary.

If the dataset has not yet been loaded, indexing returns:

```python
(None, None)
```

---

## Dataset Length

The number of metadata entries currently loaded can be obtained with `len()`:

```python
len(dataset)
```

Before the dataset has been initialized, its length is `0`.

---

## Rumble Filtering

Arden annotations are filtered before clips are extracted.

Only entries satisfying all of the following conditions are retained:

* `call_type` is `RUM` or `BKG`
* `earflap` is `0` or `1`
* `overlap` is `N`
* `quality` is `0`, `2`, `3`, or `4`
* duration is greater than `2` seconds

When `rumble_only=True`, background-noise examples are not generated.

---

## Background-Noise Generation

When `rumble_only=False`, background-noise (`BKG`) examples are automatically generated from regions outside the annotated event ranges.

The duration of generated noise clips is based on the mean and standard deviation of annotation durations in the corresponding recording.

Generated background-noise entries use:

```text
call_type = BKG
quality   = 0
overlap   = N
earflap   = 0
```

Because background-noise selection is randomized, use `random_state` when reproducible dataset generation is required.

```python
features, labels = dataset.load_data(
    random_state=123
)
```

---

## Audio Processing

Audio recordings are loaded using `librosa` at the requested sample rate:

```python
sample_rate=2000
```

For each retained metadata entry:

1. The corresponding time range is extracted from the recording.
2. A `0.2` second buffer is added to each side.
3. The clip is passed through `a3em.utils.preprocess`.
4. Acoustic features are calculated using `a3em.utils.extract_features`.

The original extracted clip and its computed features are retained by the dataset instance.

---

## Reloading Data

Once clips and features have been generated, the `Arden` instance reuses them.

To force the dataset to regenerate its metadata, clips, and features:

```python
features, labels = dataset.load_data(
    reload=True
)
```

This is useful when changing parameters such as:

```python
sample_rate
rumble_only
random_state
```

---

## Quick Start

```python
import os
from a3em.datasets import Arden

dataset = Arden(
    path=os.getenv("DATA_PATH"),
    token=os.getenv("API_TOKEN")
)

# Load features and labels
features, labels = dataset.load_data(
    random_state=123
)

print(features.head())
print(labels.head())

# Create a machine-learning split
x_train, x_test, y_train, y_test = dataset.load_data_ml(
    test_split=0.2,
    random_state=123,
    shuffle=True
)

# Access raw clips and features
clips, clip_features = dataset.load_clips()

# Iterate through clips
for clip, feature_set in dataset:
    print(clip.shape)
    print(feature_set)
```
