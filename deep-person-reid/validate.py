import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import pickle
import scipy.io
import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchreid import transforms as T
from torchreid import models
from torchreid import data_manager
from PIL import Image
from torch.autograd import Variable
from model import ft_net

parser = argparse.ArgumentParser(description='For writing the validation features')

parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
args = parser.parse_args()

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class Dataset(Dataset):
    def __init__(self, path, transform):
        self.dir = path
        self.image = [f for f in os.listdir(self.dir) if f.endswith('png')]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = self.image[idx]

        img = read_image(os.path.join(self.dir, name))
        img = self.transform(img)

        return {'name': name.replace('.png', ''), 'img': img}

def load_network(network):
    print(type(network))
    save_path = os.path.join('./model','net_last.pth')
    print(save_path)
    network.load_state_dict(torch.load(save_path))
    return network

def extractor(model, dataloader):
    def fliplr(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    test_names = []
    test_features = torch.FloatTensor()
    print(type(dataloader))
    for batch, sample in enumerate((dataloader)):
        names, images = sample['name'], sample['img']
        with torch.no_grad():
            ff = model(Variable(images.cuda())).data.cpu()
            ff = ff + model(Variable(fliplr(images).cuda())).data.cpu()
            ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))
            test_names = test_names + names
            test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features
#test transform is 
def main():
    transform_test = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("I am in the main")
    use_gpu = torch.cuda.is_available()
    #model = torch.load(os.path.join('./model','best_model.pth.tar'))
    #print(type(model))
    #model.eval()
    #model = models.init_model(name=args.arch, num_classes=751, loss={'xent'})
    #if use_gpu:
    #    model = nn.DataParallel(model).cuda()
    #checkpoint = torch.load(os.path.join('./model','net_last.pth'))
    #save_path = os.path.join('./model','net_last.pth')
    #model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(torch.load(save_path))
    #model = torch.load(os.path.join('./model','net_last.pth'))
    #if use_gpu:
    #    model = nn.DataParallel(model).cuda()
    #model.to(device)
    #model_structure = models.init_model(name=args.arch, num_classes=751, loss={'xent'})
    model_structure = ft_net(751)
    model=load_network(model_structure)
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    for subset in ['query', 'gallery']:
        print(subset)
        test_names, test_features = extractor(model, DataLoader(Dataset(subset,transform=transform_test)))
        results = {'names': test_names, 'features': test_features.numpy()}
        scipy.io.savemat(os.path.join('log_dir', 'feature_val_%s.mat' % (subset)), results)


if __name__ == "__main__":
    main()

