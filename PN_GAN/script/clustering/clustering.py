from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import os,sys
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Function to load image and give its tensor form
#imageList stores the name of the image, Later after clustering, this will be used to get the cannonical pose
def load_image(image_path, imageList, transform=None):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)
    
    imageList.append(image_path)
    if transform:
        image = transform(image).unsqueeze(0)
    
    return image.to(device)


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def main(config):
    
    # Image preprocessing
    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    # We use the same normalization statistics here.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])
    
    #This path contains all the pose images generated
    #here the current directory will be script
    images_path = os.listdir(os.path.join(os.getcwd(),'/poses'))
    Z = []
    imageList = []
    for im in images_path:
        content = load_image('/scratch/user/akhila1012/Person-Re-Identification/PN_GAN/script/poses/'+im, imageList, transform)
        vgg = VGGNet().to(device).eval()
        #Gives the feature map
        content_features = vgg(content)
        #Need to convert to CPU before converting them to numpy arrays
        b = content_features[0].to(torch.device("cpu"))
        
        #convert to numpy for the clustering task
        feature = b.squeeze().detach().numpy()
        
        #the feature vector obtained from vgg has a dimension (64,128,64). Flattening it to a vector, and appending feature per person to the list Z.
        feature = feature.flatten()
        
        #Z will be a list, where columns represent the features, and each row represents one image
        Z.append(feature)
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    print("###############Starting the clustering###########################")
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    #closest will have the image closest to each cluster center in the feature space
    closest, closest_distances = pairwise_distances_argmin_min(center, Z)
    
    #cannonical pose images will be stored in the path cannonical_poses
    if not os.path.exists('cannonical_poses'):
        os.makedirs('cannonical_poses')
    j=1
    for index in closest:
        image = cv2.imread(imageList[index])
        out_name = 'cannonical_poses/'+'cannonical_'+str(j)+'.png'
        cv2.imwrite(out_name,image)
        j=j+1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    main(config)
