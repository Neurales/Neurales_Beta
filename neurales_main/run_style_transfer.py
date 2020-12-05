# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:26:26 2020

@author: XZ-WAVE
"""

import argparse
from neurales_CV import *
from neurales_style_transfer import *


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Command Line Style Transfer')
    parser.add_argument('--content_img',type=str,default='./lana_del_rey.jpg',help='Path to content image')
    parser.add_argument('--style_img',type=str,default='./starry_nights.jpg',help='path to style image')
    parser.add_argument('--epochs',type=int,default=400,help='number of epochs to train the model for')
    parser.add_argument('--style_weight',type=int,default=10000,help='weight to control the level of styling used')
    parser.add_argument('--img_size',type=int,default=400,help='larger images take significantly more time to train on')
    parser.add_argument('--out_size',type=int,default=1000,help='resized image')
    
    args = parser.parse_args()
    
    style_img = image_loader(args.style_img,img_size=(args.img_size,args.img_size))
    content_img = image_loader(args.content_img,img_size=(args.img_size,args.img_size))

    style_transfer_img = run_style_transfer(cnn=vgg13, 
                   normalization_mean=normalization_mean, 
                   normalization_std=normalization_std,
                     content_img=content_img, 
                       style_img=style_img, input_img=content_img.clone(),
                       num_steps=args.epochs,
                       style_weight=args.style_weight, content_weight=1)

def plot_img(img,out_size=args.out_size):
    im = F.interpolate(img,size=out_size)
    im = img[0]
    im = im.permute(1,2,0).detach().numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(im)
    plt.axis('off')
    plt.savefig("style_transfer.png")
    plt.show()
    
plot_img(style_transfer_img)