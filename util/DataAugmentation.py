#!/usr/bin/env python
# encoding: utf-8

# Data Augmentation class which is used with DataLoader
# Assume numpy array face images with B x C x H x W  [-1~1]

import torch 
import scipy as sp
import numpy as np
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import pdb
from PIL import Image
import numpy as np
import scipy.io as scio

class mydataset(Dataset):
    def __init__(self, args, transform=None):
        fp = open(args.datatxt)
        lines = fp.readlines()

        self.imagepaths = []
        self.labels = []

        self.transform = transform
        for line in lines:
            array = line.split(' ')
            self.imagepaths.append(array[0])
            self.labels.append(int(array[1]))

        self.people_num = max(self.labels)


    def __getitem__(self, index): 
        imgpath, target = self.imagepaths[index], self.labels[index]
        image = Image.open(imgpath+'.jpg')
        
        img = np.array(image, dtype=np.float32)
        img = img/127.5 -1
        img = img[:,:,[2,1,0]]
        label_vector = np.zeros(self.people_num, dtype=np.float)
        label_vector[target-1] = 1.0
        pose = scio.loadmat(imgpath+'.mat')['Pose_Para'][0]
        print(label_vector)
        print(pose)
        
        
        if self.transform:
            img = self.transform(img)
        
#        transform_list += [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
#        im = transforms.Compose(transform_list)(img)

        return [im, label_vector, pose]

    def __len__(self):
        return len(self.imagepaths)

class FaceIdPoseDataset(Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, images, IDs, poses, transform=None):

        self.images = images
        self.IDs = IDs
        self.poses = poses
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        ID = self.IDs[idx]
        pose = self.poses[idx]
        if self.transform:
            image = self.transform(image)


        return [image, ID, pose]


class Resize(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        pad_width = int((new_h - image.shape[1]) / 2)
        resized_image = np.lib.pad(image, ((0,0), (pad_width,pad_width),(pad_width,pad_width)), 'edge')

        return resized_image


class RandomCrop(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_image = image[:, top:top+new_h, left:left+new_w]

        return cropped_image
