# mxnet-cpp-insightface: An easy deployable face recognition pipeline with mxnet cpp framework
There are many awesome projects show how to train and design face recognition pipeline with python(like insightface), this project show how to deploy pre-trained model to production environment with mxnet-cpp. Compare with original python version, our implement has some optimization and speed improvement around 1/3.
### dependency lib
    Mxnet and opencv library 
## How to run
### Windows
    1. Check out this project and add all src files to your vs project.
    2. Add model folder and image folder to your vs project
    3. Compile and run the project.
This project implement face recognition pipeline with mxnet c++. The whole framework contains: 
### 1. Face detect with MTCNN
    
#### 1) 
    
#### 2) 
    
### 2. Face alignment with similarity transformation


### 3. Face feature extraction(insightface or other approach)
#### 1) 

#### 2) 


## To be done
