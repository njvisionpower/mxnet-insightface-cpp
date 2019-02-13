# mxnet-cpp-insightface
This project implement an easy deployable face recognition pipeline with mxnet cpp framework.There are some awesome projects aim to train and design face recognition pipeline with python(like insightface), this project show how to deploy the pre-trained model to real production environment with mxnet-cpp. Compare with original python version, our implement has some optimization and speed improvement around 1/3.
## Dependency lib
    Mxnet and opencv library 
## How to run
### Windows
    1. Check out this project and add all src files to your vs project.
    2. Add model folder and image folder to your vs project
    3. Compile and run the project.
## Framework
This project implement face recognition pipeline with mxnet c++, and currently mainly optimize on CPU. The whole framework contains: 
### 1. Face detect with MTCNN
MTCNN is a cascade network with PNet, RNet and ONet. The first stage will sample with image pyramid.
    1.Assume that most task get images from camera so the input size is fiexed, thus the number of scales and scale size for every image is also fixed, we can initiate number=scales predict handler to avoid frequently resize or reload which will cause much time overhead.
    2.The created predict handler for different scales can be easily implement with multi-thread to make speed up.
    
#### 2) 
    
### 2. Face alignment with similarity transformation
The alignment algorithm is check out from:[face alignment with similarity transformation](https://github.com/deepinsight/insightface/blob/master/cpp-align/FacePreprocess.h)
Detail algorithm please see paper "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573.


### 3. Face feature extraction(insightface or other approach)
#### 1) 

#### 2) 


## To be done
    Optimization for different batch to feed RNet, ONet and feature extract network. Batch of input number of images will save time than loop. For example the input number x channels x width x height will get speed up than 1 x channels x width x height with loop.
