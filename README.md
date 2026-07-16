# A3EM Package
```python
import a3em
```

## a3em.utils
**preprocess**(*audio*, *sample_rate*, *normalization=0.7*)

This function takes in an audio clip as a `numpy.ndarray`, applies a high pass filter and a low pass filter, and normalizes the signal.

```python
import librosa
from a3em.utils import preprocess

audio_path = 'test.wav'
audio, sample_rate = librosa.load(audio_path)

pre_processed_audio = preprocess(audio, sample_rate)
```

**extract_features**(*audio*, *sample_rate*)

This function takes in an audio clip as a `numpy.ndarray` and returns a dictionary containing the peak frequency in the signal, the centroid of the signal, the bandwidth of the signal, the 5% and 95% frequencies of the signal, and the Mel-Frequency Cepstral Coefficients of the signal

```python
import librosa
from a3em.utils import preprocess, extract_features

audio_path = 'test.wav'
audio, sample_rate = librosa.load(audio_path)

pre_processed_audio = preprocess(audio, sample_rate)
features = extract_features(pre_processed_audio, sample_rate)
```

## a3em.datasets
This module give access to our public data sets. Information on each data set can be found below.

### a3em.datasets.arden
Collar-borne AudioMoth recordings from Arden deployed in June 2025 within Samburu National Reserve, Kenya.

**load_data**(*path=DEFAULT_PATH*, *test_split=0.25*, *random_state=123*, *shuffle: bool = True*)

This function returns a tuple containing a `pandas.DataFrame` containing the audio features of each clip and a `pandas.Series`
containing labesls.

```python
from a3em.datasets import arden

path = '/myfiles/desired_location'

(x_train, y_train), (x_test, y_test) = arden.load_data(path, test_split=0.2, random_state=123)

# if test_split is set to zero, no split is done
X, y = arden.load_data(path, test_split=0.0)
```

**load_clips**(*path=DEFAULT_PATH*, *rumbles_only=False*, *noise_seed*)

This function returns a tuple containing a list containing every audio clip stored as an `numpy.array` and a `pandas.DataFrame`
containing metadata on the clips. 

```python
from a3em.datasets import arden

path = '/myfiles/desired_location'

clips, metadata = arden.load_clips(path)

# the setting the rumbles_only field to True will only return clips of confirmed rumbles
rumbles, rumbles_metadata = arden.load_clips(path, rumbles_only=True)
```