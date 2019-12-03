# NeoNav
This is the implementation of our AAAI 2020 paper `NeoNav: Improving the Generalization of Visual Navigation via Generating Next Expected Observations`, training and evaluation on Active Vision Dataset (depth only).<br>
## Navigation Model
![](https://github.com/wqynew/NeoNav/raw/master/image/overview.png)
## Implementation
### Training
* The environment: Cuda 10.0, Python 3.6.4, PyTorch 1.0.1 
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
    <thead>
        <tr>
            <th>Start</th>
            <th>End</th>
            <th>Start</th>
            <th>End</th>
        </tr>
    </thead>
    <tbody>
       <tr>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s1.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t1.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s3.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t3.png'></td>
       </tr>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_011_1_001110011030101_001110005720101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310002970101_001310004330101.gif'></td>
       </tr>
       <tr>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s2.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t2.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/s4.png'></td>
         <td align="center"><img src='https://github.com/wqynew/NeoNav/raw/master/image/t4.png'></td>
       </tr>
       <tr>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_013_1_001310007440101_001310000150101.gif'></td>
         <td align="center" colspan=2><img src='https://github.com/wqynew/NeoNav/blob/master/image/Gif-Home_016_1_001610000060101_001610004220101.gif'></td>
       </tr>
    </tbody>
  </table>
</div>

## Contact
To ask questions or report issues please open an issue on the [issues tracker](https://github.com/wqynew/NeoNav/issues).
## Citation
If you use NeoNav in your research, please cite the paper:
```
@article{wuneonav,
  title={NeoNav: Improving the Generalization of Visual Navigation via Generating Next Expected Observations},
  author={Wu, Qiaoyun and Manocha, Dinesh and Wang, Jun and Xu, Kai}
}
```


