# Acoustic Feature Extraction from African Elephant Rumbles
DATASET

Collar-borne AudioMoth recordings from Arden deployed in June 2025 within Samburu National Reserve, Kenya

GOALS

1) Extract acoustic features from rumble vocalizations that are relevant to rumble detection/classification.
   1) Can look at Pardo et al. 2024 to see how they did feature extraction: https://doi.org/10.1098/rsos.241264. There is open source code from this publication here: https://datadryad.org/dataset/doi:10.5061/dryad.rv15dv4dt/.
   2) Determine if calls with 'quality' label of 2 are sufficient for data upload
3) Plot a histogram for each extracted feature
4) Test which extracted features best identify rumbles from background noise by comparing with randomly extracted clips of similar duration
5) Cross correlation to identify and remove correlated features

### Formatting your .env file
```bash
ANNOTATION_PATH=
AUDIO_PATH=
FIG_PATH=
PREFETCH_PATH=
```

### Setup Instructions
```bash
# set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip3 install -r requirements.txt

# set up environment variables
source .env

# deactivate virtual environment when complete
deactivate
```

### Task Log
- [ ] Evaluate other audio features to add to the extraction.
- [x] Package preprocessing and feature extraction into reusable modules or classes.
- [ ]  Test which extracted features best identify rumbles from background noise
   - [ ] Create another DataFrame using random clips from the recordings.
   - [ ] Label the clips
   - [ ] Compare feature distributions
   - [ ] Statistical testing
   - [ ] TRain a simple classifier
   - [ ] Measure feature importance
   - [ ] Visualize the feature space
   - [ ] Evaluate indicidual features