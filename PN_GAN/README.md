# PN_GAN
We have codes for GAN training and clustering the pose images into 8 clusters to get 8 cannonical pose images here.

# Clustering
Using OpenPose library, we extracted pose images with 18 pose points for every image in the training dataset of Market1501. We then obtained a set of 8 cannonical poses which are representative of the typical viewpoint and body-configurations exhibited by people. 
We used VGG-19 pre-trained on the ImageNet ILSVRC-2012 dataset to extract the features of each pose images, and K-means algorithm
is used to cluster the training pose images into canonical poses. For feature extraction, we used features extracted at the 10th layer of VGG-19 which are of dimension (256,32,16). The images closest to each cluster center are taken as the cannonical pose images.

How to run it:

&ensp;&ensp;(1) run 'clustering/clustering.py' to obtain the cannonical pose images, and these images will be saved in ./cannonical_poses folder. The clustering/clustering.py expects to have all the pose images on training data set in the directory poses/;

# Result on Market1501

<p align="center">
<figure align="center">
  <img src="https://github.com/aditya30394/Person-Re-Identification/blob/master/PN_GAN/CannonicalPoses.PNG">
  <figcaption>Canonical Poses</figcaption>
</figure>  
  <br/>
</p>

# GAN

How to run it:

&ensp;&ensp;(1) run 'GAN/train.py' to train the GAN model. The model and log file will be saved in folder 'GAN/model' and 'GAN/log' respectively. The validate images will be synthesized in 'GAN/images';

&ensp; or (2) run 'GAN/evaluate.py' to generate images for specific testing image. The output will be saved in folder 'GAN/test'

# Getting poses using the executable

<code>
.\bin\OpenPoseDemo.exe --image_dir .\bin\image --face --hand --write_images .\bin\poses --write_images_format png -disable_blending true -display 0
</code>


 

