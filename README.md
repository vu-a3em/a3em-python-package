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

**load_data**(*test_split*, *seed*)

This function returns a tuple containing a list of the individual audio clips used in our analysis and a `pandas.DataFrame` containing the audio features of each clip.
```python
from a3em.datasets import arden

(x_train, y_train), (x_test, y_test) = arden.load_data(test_split=0.2, seed=123)
```