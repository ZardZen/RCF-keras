# RCF-keras
Richer Convolutional Features for Edge Detection  
This is a keras implementation of RCF based on VGG. The network is almost same as that in the paper Richer Convolutional Features for Edge Detection, but I replaced the deconvolutional layers with subpixel convolutional layers. You can choose to use deconvs as well.  
## Data Preprocess  
You can run data_preprocess.py to create your own datasets using Canny edge detector and then run data_generate.py to generate .npy files as train sets.  
## Train  
python train.py  
## Test
python test.py  
## Reference  
[RCF-keras](https://github.com/fupiao1998/RCF-keras)  
[Keras_HED](https://github.com/lc82111/Keras_HED)
