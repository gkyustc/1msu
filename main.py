#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from model import single_DR_GAN_model as single_model
from model import multiple_DR_GAN_model as multi_model
from util.create_randomdata import create_randomdata
from train_single_DRGAN import train_single_DRGAN
from train_multiple_DRGAN import train_multiple_DRGAN
from Generate_Image import Generate_Image
import pdb

import glob

from skimage import io, transform
from matplotlib import pylab as plt

from tqdm import tqdm



class Resize(object):
    #  assume image  as H x W x C numpy array
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * w / h)
        else:
            new_h, new_w = int(self.output_size * h / w), self.output_size
        
        resized_image = transform.resize(image, (new_h, new_w))

        if h > w:
            diff = self.output_size - new_w
            if diff % 2 == 0:
                pad_l = int(diff / 2)
                pad_s = int(diff / 2)
            else:
                pad_l = int(diff / 2) + 1
                pad_s = int(diff / 2)

            padded_image = np.lib.pad(resized_image, ((0, 0), (pad_l, pad_s), (0, 0)), 'edge')

        else:
            diff = self.output_size - new_h
            if diff % 2 == 0:
                pad_l = int(diff / 2)
                pad_s = int(diff / 2)
            else:
                pad_l = int(diff / 2) + 1
                pad_s = int(diff / 2)

            padded_image = np.lib.pad(resized_image, ((pad_l, pad_s), (0, 0), (0, 0)), 'edge')

        return padded_image

def DataLoader(data_place):
    image_dir = data_place
    rsz = Resize(110)

    Indv_dir = []
    for x in os.listdir(image_dir):
        if os.path.isdir(os.path.join(image_dir, x)):
            Indv_dir.append(x)

    Indv_dir = np.sort(Indv_dir)

    images = np.zeros((7000, 110, 110, 3))
    id_labels = np.zeros(7000)
    pose_labels = np.zeros(7000)
    count = 0
    gray_count = 0

    for i in tqdm(range(len(Indv_dir))):
        Frontal_dir = os.path.join(image_dir, Indv_dir[i], 'frontal')
        Profile_dir = os.path.join(image_dir, Indv_dir[i], 'profile')

        front_img_files = os.listdir(Frontal_dir)
        prof_img_files = os.listdir(Profile_dir)

        for img_file in front_img_files:
            img = io.imread(os.path.join(Frontal_dir, img_file))
            
            if len(img.shape) == 2:
                gray_count = gray_count + 1
                continue
            img_rsz = rsz(img)
            
            images[count] = img_rsz
            id_labels[count] = i
            pose_labels[count] = 0
            count = count + 1

        for img_file in prof_img_files:
            img = io.imread(os.path.join(Profile_dir, img_file))
            if len(img.shape) == 2:
                gray_count = gray_count + 1
                continue
            img_rsz = rsz(img)
            images[count] = img_rsz
            id_labels[count] = i
            pose_labels[count] = 1
            count = count + 1

    id_labels = id_labels.astype('int64')
    pose_labels = pose_labels.astype('int64')

    # [0,255] -> [-1,1]
    images = images*2 - 1

    # RGB -> BGR
    images = images[:, :, :, [2, 1, 0]]
    # B x H x W x C-> B x C x H x W
    images = images.transpose(0, 3, 1, 2)

    # 白黒画像データを取り除く
    images = images[:gray_count * -1]
    id_labels = id_labels[:gray_count * -1]
    pose_labels = pose_labels[:gray_count * -1]
    Np = int(pose_labels.max() + 1)
    Nd = int(id_labels.max() + 1)
    Nz = 50
    channel_num = 3

    return [images, id_labels, pose_labels, Nd, Np, Nz, channel_num]




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning & saving parameterss
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=1000, help='number of epochs for train [default: 1000]')
    parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 8]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-save-freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    # data souce
    parser.add_argument('-random', action='store_true', default=False, help='use randomely created data to run program')
    parser.add_argument('-data-place', type=str, default='./data', help='prepared data path to run program')
    # model
    parser.add_argument('-multi-DRGAN', action='store_true', default=False, help='use multi image DR_GAN model')
    parser.add_argument('-images-perID', type=int, default=0, help='number of images per person to input to multi image DR_GAN')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-generate', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-datatxt', type=str, default='./imagepaths', help='the txt containing the path of the training images')

    args = parser.parse_args()

    # update args and print
    if args.multi_DRGAN:
        args.save_dir = os.path.join(args.save_dir, 'Multi',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        args.save_dir = os.path.join(args.save_dir, 'Single',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(args.save_dir)

    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text ="\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
            f.write(text)


    # input data
    if args.random:
        images, id_labels, pose_labels, Nd, Np, Nz, channel_num = create_randomdata()
    else:
        print('n\Loading data from [%s]...' % args.data_place)
        try:
            images, id_labels, pose_labels, Nd, Np, Nz, channel_num = DataLoader(args.data_place)
        except:
            print("Sorry, failed to load data")

    # model
    if args.snapshot is None:
        if not(args.multi_DRGAN):
            D = single_model.Discriminator(Nd, Np, channel_num)
            G = single_model.Generator(Np, Nz, channel_num)
        else:
            if args.images_perID==0:
                print("Please specify -images-perID of your data to input to multi_DRGAN")
                exit()
            else:
                D = multi_model.Discriminator(Nd, Np, channel_num)
                G = multi_model.Generator(Np, Nz, channel_num, args.images_perID)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
          #  D = torch.load('{}_D.pt'.format(args.snapshot))
            G = torch.load('{}_G.pt'.format(args.snapshot))
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()

    if not(args.generate):
        if not(args.multi_DRGAN):
            train_single_DRGAN(images, id_labels, pose_labels, Nd, Np, Nz, D, G, args)
        else:
            if args.batch_size % args.images_perID == 0:
                train_multiple_DRGAN(images, id_labels, pose_labels, Nd, Np, Nz, D, G, args)
            else:
                print("Please give valid combination of batch_size, images_perID")
                exit()
    else:
        # pose_code = [] # specify arbitrary pose code for every image
#        pose_code = np.random.uniform(-1,1, (images.shape[0], Np))
        pose_code = np.ones((images.shape[0],Np))
        features = Generate_Image(images, pose_code, Nz, G, args)
