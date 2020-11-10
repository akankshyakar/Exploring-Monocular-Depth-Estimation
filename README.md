# Semi-Supervised-3D-Structural-In-variance

Code for project 
Semi-Supervised 3D Structural In-variance for World Coordinates Prediction for realistic AR object placement

Recent works focusing on Indoor scenes try to predict 3D world coordinates and place objects in real world for Augmented reality applications, specially for Monocular hand held phones and devices. This is challenging because indoor scenes have walls, corridors and featureless planar objects. Height and Uprightness Invariance for 3D Prediction from a Single View [1] in CVPR 2020 tries to solve this by predicting the camera intrinsics and the camera position with respect to the ground. It then predicts the 3D position for each pixel. However the 3D structural accuracy of this method is worse compared to DORN [4] which was the earlier state of the art in monocular depth estimation. Geometry based monocular SLAM methods which work well in outdoor environments for self-driving cars, fail indoors due to large texture-less regions such as walls. In order to solve this, we borrow ideas from [5] [10] which use global and local planar constraints by modifying loss functions and as this paper performs poorly than DORN, we will modify the network to enable Local Planar guidance from BTS [6]. Finally, using this we will find the world coordinates for the system with accurate structure and project realistic placement of AR objects in indoor scenes