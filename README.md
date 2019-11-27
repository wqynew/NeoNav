# NeoNav
This is the implementation of our AAAI 2020 paper “NeoNav: Improving the Generalization of Visual Navigation via Generating Next Expected Observations”, training and evaluation on Active Vision Dataset (depth only).<br>
## Navigation Model
![](https://github.com/wqynew/NeoNav/raw/master/image/overview.png)
## Implementation
### Training
* The environment: cuda 10.0, Python 3.6.4, pytorch 1.0.1 
* The dataset: AVD_Minimal: https://storage.googleapis.com/active-vision-dataset/AVD_Minimal.zip 
* Please download "depth_imgs.npy" file from the AVD_Minimal and put the file locally. 
* Please download our training data [HERE] (https://drive.google.com/open?id=1Avl5CNn-V4Fpfhn0nE9siJMkYZRczKmN).
* The pretrained models can be download from [HERE](https://drive.google.com/open?id=182D_0hP7orpJKyDDLlUyV4URwT3Rt0Ux). If you plan to train your own navigation model from scratch, some suggestions are provided:


