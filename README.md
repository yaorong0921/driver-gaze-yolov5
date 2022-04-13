## Overview
The code for saliency metrics is based on: https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py
The code vor the convolutional LSTM is based on: https://github.com/yaorong0921/Driver-Intention-Prediction/blob/master/models/convolution_lstm.py
The code for extracting YOLOv5 features is based on:  https://github.com/ultralytics/yolov5 (Release 5.0)

### Training and Testing
Our model can be trained and tested with the following steps:
    1. Prepare BDDA images:
    Download videos from https://bdd-data.berkeley.edu/ and extract images
    Camera image names are expected as videoNr_imgNr and corresponding gaze maps as videoNr_pure_hm_imgNr.

    2. Extract features:
    Download weights (yolov5s.pt) and code from https://github.com/ultralytics/yolov5 (Release 5.0) and run extract_features.sh
    (extract_features.py is a modified version of detect.py)

    3. Compute the ground-truth grids and save them in a txt-file:
    Run compute_grid.sh

    4. Train and test:
    For gaze map prediction and pixel-level/object-level evaluation run gaze_prediction_and_evaluation.sh
    Training is optional, checkpoints for our trained grid 16x16 models (without LSTM, with LSTM seqlen 8, with convLSTM seqlen 6) are available.
    Features are expected within folders with names training/validation/test.


### Misc.
More files can be found in the 'More files' directory:
    1. Scripts for computing and evaluating the baseline (average of all BDD-A training gaze maps)
    2. The script that we used to compute the model complexity.
    3. A script for evaluating other models based on given predicted gaze maps.
    4. A script for drawing the ROC curves and computing the optimal thresholds.
