## Overview
The code for saliency metrics is based on: https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py   
The code for the convolutional LSTM is based on: https://github.com/yaorong0921/Driver-Intention-Prediction/blob/master/models/convolution_lstm.py   
The code for extracting YOLOv5 features is based on:  https://github.com/ultralytics/yolov5 (Release 5.0).  

### Training and Testing
Our model can be trained and tested with the following steps:

1. Prepare BDD-A images:  
Download videos from https://bdd-data.berkeley.edu/ and extract images.   
Camera image names are expected as videoNr_imgNr and corresponding gaze maps as videoNr_pure_hm_imgNr.

2. Extract features and save object bounding boxes:  
Download weights (yolov5s.pt) and code from https://github.com/ultralytics/yolov5 (Release 5.0) and run extract_features.sh
(extract_features.py is a modified version of detect.py)

3. Compute the ground-truth grids and save them in a txt-file:  
Run compute_grid.sh

4. Train and test:  
For gaze map prediction and pixel-level/object-level evaluation run gaze_prediction_and_evaluation.sh.  
Training is optional, checkpoints for our trained grid 16x16 models (without LSTM, with LSTM and sequence length 8, with convLSTM and sequence length 6) are available.      
Features are expected within folders with names training/validation/test.   


Possible folder structure to test our pretrained 16x16 model without LSTM:  
```
project/  
├── bdda.py   
├── network.py   
├── gaze_prediction_and_evaluation.py   
├── grid616_model_best.pth.tar  
├── BDDA/   
│   └── test/   
│       └── gazemap_images/  # Ground-truth gaze map images (Created in step 1)  
│           ├── 1_pure_hm_00000.jpg  
│           ├── 1_pure_hm_00333.jpg  
│           ├── 1_pure_hm_00666.jpg  
│           └── ...  
├── features/  # Extracted YOLOv5 features (Created in step 2)  
│   └── test/  
│       ├── 1_00000.pt  
│       ├── 1_00333.pt  
│       ├── 1_00666.pt  
│       └── ...  
├── grids/  
│   └── grid1616/  
│       └── test_grid.txt   # Grids of test set images (Created in step 2)  
├── yolo5_boundingboxes/  # Detected YOLOv5 object bounding boxes (Created in step 2)  
│   ├── 1_00000.txt  
│   ├── 1_00333.txt  
│   ├── 1_00666.txt  
│   └── ...  
└── results/   
    └── grid1616/  # Folder to save predicted gaze maps     
```



### Misc.
More files can be found in the 'More files' directory:  
1. Scripts for computing and evaluating the baseline (average of all BDD-A training gaze maps)  
2. The script that we used to compute the model complexity.  
3. A script for evaluating other models based on given predicted gaze maps.  
4. A script for drawing the ROC curves and computing the optimal thresholds.  


### Citation
If you find this work useful or use the code, please cite as follows:

```
@article{rong2022and,
  title={Where and What: Driver Attention-based Object Detection},
  author={Rong, Yao and Kassautzki, Naemi-Rebecca and Fuhl, Wolfgang and Kasneci, Enkelejda},
  journal={arXiv preprint arXiv:2204.12150},
  year={2022}
}
```
