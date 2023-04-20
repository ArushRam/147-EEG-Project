# EEG Classifier

This project built and compared various neural network architectures for classifying electro-encephalogram (EEG). The dataset used was [BCI Competition IV 2a](https://lampz.tugraz.at/~bci/database/001-2014/description.pdf) (Brunner, Clemens, et al.).

### Architectures

We tested basic CNN, CNN-LSTM and Inception-module architectures, achieving the best performance with a single Inception module (~75%). The architectures used can be found in the ```networks``` folder. The basic CNN architecture was largely used as a test-bench for identifying best-performing filter sizes which we then incorporated into the inception module.

### Data Augmentation

We cropped to include only the first 512 data points of the signal as these were most informative. We then performed random crops to amplify the training data.

We also removed noise by applying a 4th-order butterworth filter with threshold 60Hz. One technique we did not have time to explore was mixing higher-frequency and lower-frequency components of different samples, which was used in some research papers to positive effect.

### References

1. Brunner, Clemens, et al. "BCI Competition 2008–Graz data set A." Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology 16 (2008): 1-6.


#### Note

Data to be stored in "data" folder in root directory.