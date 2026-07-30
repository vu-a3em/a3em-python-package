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
This module give access to our public data sets. Information on each data set can be found below.

## Arden
Collar-borne AudioMoth recordings from Arden deployed in June 2025 within Samburu National Reserve, Kenya.

### load_data
Arden.**load_data**(api_token, path, test_split=0.2, random_state=None, sample_rate=2000, rumble_only=False, shuffle=False)

This function returns tuples containing a `pandas.DataFrame` containing the audio features of each clip and a `pandas.Series`
containing labesls. 

**Parameters**
- `api_token`: This is the token that can be accected through the user's Dryad account. It is necessary to access the database.
- `path`: The path where local data should be stored.
- `test_split`: Should be a float between 0.0 and 1.0. This is the fraction of data points reserved for testing.
- `random_state`: The seed used for all psuedo-random properties of the class.
- `sample_rate`: The desired sample rate for audio data
- `rumble_only`: Whether to only include elephant rumbles in the dataset or also include background noise.
- `shuffle`: Whether or not to shuffle the data before splitting.

```python
from a3em.datasets import Arden
import os

path = os.getenv('DATA_PATH')
token = os.getenv('API_TOKEN')

(x_train, y_train), (x_test, y_test) = Arden.load_data(token, path, test_split=0.2, random_state=123)
```

### load_clips
Arden.**load_clips**(api_token, path, rumble_only=False, random_state=None, sample_rate=2000)

This function returns a tuple containing a list containing every audio clip stored as a `numpy.array` and a `pandas.DataFrame`
containing metadata on the clips. 

**Parameters**
- `api_token`: This is the token that can be accected through the user's Dryad account. It is necessary to access the database.
- `path`: The path where local data should be stored.
- `rumble_only`: Whether to only include elephant rumbles in the dataset or also include background noise.
- `random_state`: The seed used for all psuedo-random properties of the class.
- `sample_rate`: The desired sample rate for audio data

```python
from a3em.datasets import arden
import os

path = os.getenv('DATA_PATH')
token = os.getenv('API_TOKEN')

clips, metadata = arden.load_clips(token, path)

# the setting the rumbles_only field to True will only return clips of confirmed rumbles
rumbles, rumbles_metadata = arden.load_clips(token, path, rumbles_only=True)
```

### iter
Arden.**iter**(api_token, path rumble_only=False, random_state=None, sample_rate=2000)

This is a generater function that iterates over each value from **load_clips**. Each element returned is a tuple containing an audio
clip stored as a `numpy.array` and a dictionary containing the metadata for the clip.

**Parameters**
- `api_token`: This is the token that can be accected through the user's Dryad account. It is necessary to access the database.
- `path`: The path where local data should be stored.
- `rumble_only`: Whether to only include elephant rumbles in the dataset or also include background noise.
- `random_state`: The seed used for all psuedo-random properties of the class.
- `sample_rate`: The desired sample rate for audio data

```python
from a3em.datasets import arden
import os

path = os.getenv('DATA_PATH')
token = os.getenv('API_TOKEN')

arden_iter = Arden.iter(token, path)
clip, metadata = next(arden_iter)
```