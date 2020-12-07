# Exploring-Monocular-Depth-Estimation

### Command
```python3 train.py -m 'rgb' --save-path "<path to save>" --path "<data for NYU>"```

![alt text](https://github.com/akankshyakar/Exploring-Monocular-Depth-Estimation/blob/main/images/result.png?raw=true)


Most of our code is borrowed from Github implementation of other papers. Our main motivation was to explore depth estimation using NYU RGBD data. We made the network to be both Regressor and a Classifier. We added [LPG](https://github.com/cogaplex-bts/bts/tree/master/pytorch), [VNL Loss](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction), [Ordinal Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html)  and [Photo Metric Loss](https://github.com/ClementPinard/SfmLearner-Pytorch). 


Our PPT can be found [here](https://docs.google.com/presentation/d/1hJW_AYzt0oAWWwNoiXRbCJRGS_835kfl62t-LIyp4pw/edit?usp=sharing) and project report and in depth results [here](https://www.google.com).


### TODO:

- [x] Overall pipeline - AK
- [x] DispNet - AK
- [X] PoseNet - AB
- [x] VNL Loss - AK
- [x] LPG constraint - AK
- [x] Photometric Reconstruction - AB
- [x] Height & uprightness Loss -AB (don't work though)
- [x] Validation - AK
- [x] Accuracy Metrics - AK
- [X] Visualize Tensorboard - AK
- [X] Ordinal Regression (SORD) - AB


### EXPERIMENTS:

- [x] Only VNL, LPG (AK) ``` python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/new_save/vnlonlylpg" --lpg  --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0 --vnl-loss 1 --l1 0 --ordinal 0 --epochs 15 ```
- [x]  Only VNL,  NO LPG (AK) ```python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/new_save/vnlnolpg"  --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0 --vnl-loss 1 --l1 0 --ordinal 0 --epochs 15 ```
- [x] VNL 0.5 photo 0.5 LPG (AK) ```sudo python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/new_save/vnl5photo5lpg" --lpg  --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0.5 --vnl-loss 0.5 --l1 0 --ordinal 0 --epochs 15```
- [x] VNL 0.5 photo 0.5  NO LPG (AK) ```sudo python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/new_save/vnl5photo5nolpg" --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0.5 --vnl-loss 0.5 --l1 0 --ordinal 0 --epochs 15```
- [x] VNL 0.75 photo 0.25  NO LPG (AK) CUDA_VISIBLE_DEVICES="1" sudo python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/saving/vnl75photo25nolpg" --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0.25 --vnl-loss 0.75 --l1 0 --im2pcl 0 --epochs 20
- [x] VNL 0.75 photo 0.25 LPG (AK) CUDA_VISIBLE_DEVICES="1" sudo python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/saving/vnl75photo25lpg" --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0.25 --vnl-loss 0.75 --l1 0 --im2pcl 0 --epochs 20
- [x] VNL 0.5 ordinal 0.5 (AB) python train.py  --save-path "/home/rpl/abhorask/geoviz/Semi-Supervised-3D-Structural-In-variance/saving/v5o5"   --path "/media/rpl/Data/abhorask/nyuv2/data/" --epoch-size 5000 --epochs 20 --l1 0 --photometric 0 --vnl-loss 0.5  --ordinal 0.5 --gpu_id 0
- [X] VNL 0.5 ordinal 0.5 (AB) python train.py  --save-path "/home/rpl/abhorask/geoviz/Semi-Supervised-3D-Structural-In-variance/saving/v5o5"   --path "/media/rpl/Data/abhorask/nyuv2/data/" --epoch-size 5000 --epochs 20 --l1 0 --photometric 0 --vnl-loss 0.5  --ordinal 0.5 --gpu_id 0
- [X] VNL 0.5 ordinal (v3) 1 (AB) python3 train.py  --save-path "/home/rpl/abhorask/geoviz/Semi-Supervised-3D-Structural-In-variance/saving/v5o10"   --path "/media/rpl/Data/abhorask/nyuv2/data/" --epoch-size 5000 --epochs 12 --l1 0 --photometric 0 --vnl-loss 1 --ordinal 1 --max-depth 20
- [X] Only Photometric (AB)  python3 train.py  --save-path "/home/rpl/abhorask/geoviz/Semi-Supervised-3D-Structural-In-variance/saving/photometricOnly"   --path "/media/rpl/Data/abhorask/nyuv2/data/" --epoch-size 5000 --epochs 12 --l1 0 --photometric 1 --vnl-loss 0 --ordinal 0
