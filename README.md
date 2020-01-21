# Video_Captioning_using_ActionVLAD_and_Bidirectional_LSTMs

## Requirements

- skimage
- numpy
- ImageURLs.txt

Note that I only tested in python3 (maybe with some tweaks it will run in python2).

## How to Download NUS-WIDE raw image

```
python3 download_nus_wide.py
```

There are some options you can give, see more info with,

```
python3 download_nus_wide.py --help
```

## Note

NUS-WIDE dataset has 269,948 images but with this downloader it can get only 221,166 raw images.

## Task 1
Image classification using a MLFFNN with Deep CNN features for
an image as the input to the MLFFNN.

## Task 2
Image annotation using Deep CNN features as input to MLFFNN.

## Task 3 
Image captioning using a Deep CNN as encoder and a single hidden layer LSTM
based RNN as decoder.

## Task 4
Image captioning using VLAD features and LSTM.

## Task 5
Video Captioning using ActionVLAD as encoder and a single hidden
layer LSTM based RNN as decoder.
