# Semi-Supervised-3D-Structural-In-variance

Code for project 
Semi-Supervised 3D Structural In-variance for World Coordinates Prediction for realistic AR object placement

python3 train.py -m 'rgb' --save-path "../Semi-Supervised-3D-Structural-In-variance/saving" --path "/media/mscv/SecondHDD/data/nyu/"
command to run

TODO:

- [x] Overall pipeline - AK
- [x] DispNet - AK
- [X] PoseNet - AB
- [x] VNL Loss - AK
- [x] LPG constraint - AK
- [x] Photometric Reconstruction - AB
- [x] Height & uprightness Loss -AB (don't work though)
- [x] Validation - AK
- [x] Accuracy Metrics - AK
- [ ] AR objects
- [X] Visualize Tensorboard - AK


EXPERIMENTS:

- [x] Only VNL, LPG (AK) ``` python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/saving/vnlonlylpg" --lpg  --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0 --vnl-loss 1 --l1 0 --im2pcl 0 --epochs 20 ```
- [x]  Only VNL,  NO LPG (AK) ```python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/saving/vnlnolpg"  --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0 --vnl-loss 1 --l1 0 --im2pcl 0 --epochs 20 ```
- [ ] VNL 0.5 photo 0.5 LPG (AK) ```sudo python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/saving/vnl5photo5lpg" --lpg  --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0.5 --vnl-loss 0.5 --l1 0 --im2pcl 0 --epochs 20```
- [ ] VNL 0.5 photo 0.5  NO LPG (AK) ```sudo python3 train.py  --save-path "/media/mscv/SecondHDD/Semi-Supervised-3D-Structural-In-variance/saving/vnl5photo5nolpg" --path "/media/mscv/SecondHDD/data/nyu/" --epoch-size 5000  --photometric 0.5 --vnl-loss 0.5 --l1 0 --im2pcl 0 --epochs 20```
- [ ] 
- [ ] 
- [ ] 
- [ ]
- [ ] 
- [ ] 
- [ ] 

