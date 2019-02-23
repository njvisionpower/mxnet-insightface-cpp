# mxnet-insightface-cpp 

[中文](README_zh.md)

<p align="center"> 
<img src="https://github.com/njvisionpower/mxnet-insightface-cpp/blob/master/demo.jpg" width = 60% height = 60%>
</p>  

*****
This project implement an easy deployable face recognition pipeline with mxnet cpp framework.There are some awesome projects aim to train and design face recognition pipeline with python(like insightface), this project show how to deploy the pre-trained model to real production environment with mxnet-cpp. Compare with original python version, our implement has some optimization and speed improvement around 1/3.
## Dependency lib
    Mxnet and opencv library 
## Operation
#### Make labels
    Extract features with images in "images" folder, and also will generate labels
#### Face recognition with camera
    Extract features from camera image and compare distance with labeled features.
## Framework
This project implement face recognition pipeline with mxnet c++, and currently mainly optimize on CPU. The whole framework contains:  
### 1. Face detect with MTCNN
MTCNN is a cascade network with PNet, RNet and ONet. The first stage will sample with image pyramid.  
    
**1**.Assume that most task get images from camera so the input size is fiexed, thus the number of scales and scale size for every image is also fixed, we can initiate number=scales predict handler to avoid frequently resize or reload which will cause much time overhead.  
**2**.The created predict handler for different scales can be easily implement with multi-thread to make speed up.  
        
### 2. Face alignment with similarity transformation
The alignment algorithm check out from:[face alignment with similarity transformation](https://github.com/deepinsight/insightface/blob/master/cpp-align/FacePreprocess.h). It will transform detect image to norm face based on 5 average points. Detail algorithm please see paper "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573.

### 3. Face feature extraction(insightface or other approach)
Feature extraction phase utilizes deep network to extract features from aligned image, and then calculate the euclidean distance or cosine distance with labeled features. You can set threshold to do face verify, recognition or other task. In this project we show the efficient **mobileface network** to do face recognition with camera(you can change the camera io base on your device, we just set "0" in codes).


## To be done
Optimization for different batch to feed RNet, ONet and feature extract network. Batch of input number of images will save time than loop. For example the input number x channels x width x height will get speed up than 1 x channels x width x height with loop.

## Reference
1. [Insightface](https://github.com/deepinsight/insightface)  
2. [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks](https://github.com/kpzhang93/MTCNN_face_detection_alignment)  
3. [MXNET](https://github.com/apache/incubator-mxnet)  
4. Least-squares estimation of transformation parameters between two point patterns
