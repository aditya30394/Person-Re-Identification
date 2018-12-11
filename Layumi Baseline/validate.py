import os
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import pickle
import scipy.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms as T
from PIL import Image
from torch.autograd import Variable
from model import ft_net,PCB,PCB_test

parser = argparse.ArgumentParser(description='For writing the validation features')

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
    save_path = os.path.join('./model','net_59.pth')
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

    for batch, sample in enumerate((dataloader)):
        names, images = sample['name'], sample['img']
        n, c, h, w = images.size()
     
        ff = torch.FloatTensor(n,2048,6).zero_()
        ff = model(Variable(images.cuda(), volatile=True)).data.cpu()
        ff = ff + model(Variable(fliplr(images).cuda(), volatile=True)).data.cpu()
        ff = ff.div((torch.norm(ff, p=2, dim=1, keepdim=True)*np.sqrt(6)).expand_as(ff))
        ff = ff.view(ff.size(0), -1)

        test_names = test_names + names
        test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features
#test transform is 
def main():
    transform_test = T.Compose([
        T.Resize((288,144),interpolation=3),
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
    model_structure = PCB(751)
    model=load_network(model_structure)
    model = PCB_test(model)
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    for subset in ['query', 'gallery']:
        print(subset)
        test_names, test_features = extractor(model, DataLoader(Dataset(subset,transform=transform_test)))
        results = {'names': test_names, 'features': test_features.numpy()}
        scipy.io.savemat(os.path.join('example', 'feature_val_%s.mat' % (subset)), results)


if __name__ == "__main__":
    main()
