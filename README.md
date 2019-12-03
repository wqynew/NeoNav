# NeoNav
This is the implementation of our AAAI 2020 paper "NeoNav: Improving the Generalization of Visual Navigation via Generating Next Expected Observations", training and evaluation on Active Vision Dataset (depth only).<br>
## Navigation Model
![](https://github.com/wqynew/NeoNav/raw/master/image/overview.png)
## Implementation
### Training
* The environment: cuda 10.0, Python 3.6.4, pytorch 1.0.1 
* Please download "depth_imgs.npy" file from the [AVD_Minimal](https://storage.googleapis.com/active-vision-dataset/AVD_Minimal.zip) and put the file in the train folder. 
* Please download our training data [HERE](https://drive.google.com/open?id=1Avl5CNn-V4Fpfhn0nE9siJMkYZRczKmN).
* Our trained models can be download from [HERE](https://drive.google.com/open?id=182D_0hP7orpJKyDDLlUyV4URwT3Rt0Ux). If you plan to train your own navigation model from scratch, some suggestions are provided:
    * Pre-train the model by using "python3 ttrain.py" and terminate the training when the action prediction accuracy approaches 70%.
    * Use "python3 train.py" to train the NeoNav model.
    
### Testing
* To evaluate our model, please run "python3 evaluate.py" or "python3 evaluete_with_stop.py"

## Results

<div align="center">
  <table style="width:100%" border="0">
    <tr>
      <td align="center"><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
      <td align="center"><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
    </tr>
    <tr>
      <td align="center">Target: Fridge</td>
      <td align="center">Target: Television</td>
    </tr>
    <tr>
      <td align="center"><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310007440101_001310000150101.gif'></td>
      <td align="center"><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_016_1_001610000060101_001610004220101.gif'></td>
    </tr>
    <tr>
      <td align="center">Target: Microwave</td>
      <td align="center">Target: Couch</td>
    </tr>
  </table>
</div>


