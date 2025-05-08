# Image Classification using Bag-of-Visual-Word
Compare the accuracy of SVM and Naive Bayes in image classification when using Bag-of-Visual-Word model.

## BoVW Image Classification Model
### Feature Extraction
keypoints are detected from each image using the ORB (Oriented FAST and Rotated BRIEF)<br>
&nbsp;&nbsp;&nbsp; ● ORB detects keypoints and describes to using a binary vector, ensuring fast to compute<br>
A descriptor vector is computed for each keypoint to represent local visual features<br>

### Visual Vocabulary Construction
All descriptor vectors from the training images are collected and clustered using the K-means algorithm<br>
&nbsp;&nbsp;&nbsp;● Each cluster center becomes a "visual word," forming a visual vocabulary (codebook) of fixed size<br>

### Histogram Representation
Each image is converted into a histogram based on the frequency of visual words<br>
Descriptors are assigned to the nearest cluster center, and the number of assignments to each cluster is counted<br>
&nbsp;&nbsp;&nbsp; ● This results in a fixed-length feature vector representing the image (e.g., a 200-dimensional vector)<br>
The histogram vectors from training images are used to train classification models<br>

## ⬇ Installation

#### 0. Prerequisites
● Download Train, Test Dataset
https://drive.google.com/file/d/1C1uMYMlDDFEbn6BwoDOFO5l6rwuZ_0tA/view?usp=sharing


#### 1. Install requirements
```
pip install -r requirements.txt
```

#### 2. Install development requirements (Optional)
```
pip install -r requirements-dev.txt
```

### 3. Train:
```
$ python train.py
```
This command will train 2 models, one based on SVM and the other on Naive Bayes, to classify images of three classes, Airplane, Face, and Motor. The trained models will be saved in the `models` directory.
- Training data:
    - Airplane: 700 images
    - Face: 300 images
    - Motorbike: 700 images
- Test data:
    - Airplane: 100 images
    - Face: 100 images
    - Motorbike: 100 images
     
### 4. Evaluate:
```
$ python evaluate.py
```
This command will evaluate the accuracy of the two models trained in the previous step.
```text
SVM Classification Accuracy: 95.0%
Naive Bayes Classification Accuracy: 84.0%
```
