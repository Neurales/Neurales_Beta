# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:50:10 2020

@author: XZ-WAVE
"""

import argparse
from neurales_CV import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom autoencoder')
    parser.add_argument('--mode',type=str,default='standard',
                        help='standard or denoise')
    parser.add_argument('--path',type=str,help='enter in a path to training data')
    parser.add_argument('--num_conv_layers',type=int,default=3,help='number of convolutional layers')
    parser.add_argument('--channels',type=int,default=3,
                        help='number of input channels into the autoencoder')
    parser.add_argument('--optimizer',type=str,default='sgd',
                        help='optimizer choice: sgd, adam or rmsprop')
    parser.add_argument('--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--momentum',type=float,default=0.9,help='momentum')
    parser.add_argument('--img_size',type=int,default=40,help='size of input')
    parser.add_argument('--cmap',type=str,default=None,help='matplotlib color map')
    parser.add_argument('--normalize',type=bool,default=True,help='normalize images or not')
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=100,help='batch size')
    parser.add_argument('--mean',type=float,default=0.0,
                        help='mean of corrputing noise distribution (Gaussian)')
    parser.add_argument('--sigma',type=float,default=0.1,
                        help='standard deviation of corrupting noise distribution (Gaussian)')
    parser.add_argument('--explore',type=bool,default=True,
                        help='explore or debug after model runs')
    
    args = parser.parse_args()
    ae = custom_ConvAE(num_conv_layers=3,nc=3)
    if args.mode == 'standard':
        
        results = train_autoencoder(path=args.path,
                                    model=ae, 
                                    batch_size=args.batch_size, 
                                    transform='default', 
                                    epochs=args.epochs, 
                                    optimizer=args.optimizer,
                                    lr=args.lr, 
                                    momentum=args.momentum,
                                    weight_decay=0, 
                                    img_size=(args.img_size,args.img_size), 
                                    cmap=args.cmap, 
                                    normalize_plot=args.normalize) 
    if args.mode == 'denoise':
        
        results = train_denoising_autoencoder(path=args.path, #path to training images
                            model=ae, #model to use
                            batch_size=args.batch_size, #batch size
                            epochs=args.epochs, #epochs to train
                            transform='default', #default transform
                            optimizer=args.optimizer, #using Adam optimizer
                            lr=args.lr, #learning rate
                            momentum=args.momentum, #momentum
                            weight_decay=0, #weight decay
                            loss_fun='mse', #use MSE loss to train denoising autoencoder (can use MAE or KL_DIV loss instead)
                            mean=args.mean, #mean of the noise distribution to corrupt the images
                            sigma=args.sigma, #standard deviation of the noise distribution to corrupt the images
                            img_size=(args.img_size,args.img_size), #size of the images
                            cmap=args.cmap, #use a colormap when plotting decoded images
                            normalize_plot=args.normalize,
                            plot_samples=False) #normaolze the plot of decoded images or not
    if args.explore == True:
        import pdb
        pdb.set_trace()
        
        