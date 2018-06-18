Implementing the version 1 of YOLO on PyTorch

Link to YOLO: https://pjreddie.com/darknet/yolov1/

Small example of training is in the notebook testing.ipynb. The model didn't have a problem with training images (so in theory it works), but in validation the model often predicted the class 'person' in the middle. Maybe overfitting or too small batch size.
