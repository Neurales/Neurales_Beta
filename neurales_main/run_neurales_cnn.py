# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:16:20 2020

@author: XZ-WAVE
"""

import argparse
from neurales_vision_utils import *
from neurales_CV import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neurals CNN for classification')
    parser.add_argument('--train_path',type=str,help='path to training data')
    parser.add_argument('--val_path',type=str,help='path to test/validation data')
    parser.add_argument('--channels',type=int,default=3,help='3 for RGB and 1 for greyscale images')
    parser.add_argument('--num_conv_layers',type=int,default=4,help='number of convolutional layers')
    parser.add_argument('--batch_norm',type=bool,default=False,help='whether or not to use Batch Normalization')
    parser.add_argument('--dropout',type=bool,default=False,help='whether or not to use dropout')
    parser.add_argument('--adaptive_type',type=str,default='A',
                        help='A for Adaptive Average Pooling and M for Adaptive Max Pooling')
    parser.add_argument('--adaptive_size',type=int,default=5,
                        help='output size of final convolutional layer')
    parser.add_argument('--epochs',type=int,default=5,help='epochs to train the model')
    parser.add_argument('--batch_size',type=int,default=10,help='batch size')
    parser.add_argument('--lr',type=float,default=1e-3,help='learning rate for optimizer')
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum for optimizer')
    parser.add_argument('--weight_decay',type=float,default=0,help='weight decay')
    parser.add_argument('--optimizer',type=str,default='adam',help='optimizer choice')
    parser.add_argument('--img_size',type=int,default=40,help='img size (square only)')
    parser.add_argument('--explore',type=bool,default=True,
                        help='explore/debug after evolution finishes')
    
    args = parser.parse_args()
    if args.train_path in ['mnist','fashion_mnist','cifar10']:
        num_classes = 10
    if args.train_path == 'cifar100':
        num_classes = 100
    if args.val_path != None:
        num_classes = len(os.listdir(args.train_path))
    if args.adaptive_type == 'A':
        pool_type = 'avg'
    if args.adaptive_type == 'M':
        pool_type = 'max'
    model = custom_cnn(num_conv_layers=args.num_conv_layers,
                 nc=args.channels,
                 num_classes=num_classes, 
                 batch_norm=args.batch_norm, 
                 dropout=args.dropout, 
                 adaptive_pool_type=pool_type, 
                 adaptive_pool_size=args.adaptive_size,
                 conv_dim=2) #i
    results = train_cnn(train_path=args.train_path,
                        val_path=args.val_path,
                        model=model,
                        transform='default',
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        optimizer=args.optimizer,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        img_size=(args.img_size,args.img_size))
    
    if args.explore == True:
        import pdb
        pdb.set_trace()