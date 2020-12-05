# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:00:36 2020

@author: XZ-WAVE
"""

import argparse
from neurales_CV import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom autoencoder')
    parser.add_argument('--path',type=str,help='path to train data')
    parser.add_argument('--num_conv_layers',type=int,default=3,
                        help='number of convolutional layers')
    parser.add_argument('--channels',type=int,default=3,
                        help='number of input channels for image')
    parser.add_argument('--noise_dim',type=int,default=3,
                        help='number of noisy channels to feed in the Generator')
    parser.add_argument('--img_size',type=int,default=40)
    parser.add_argument('--epochs',type=int,default=25,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=10,help='batch size')
    parser.add_argument('--d_opt',type=str,default='sgd',help='optimizer for Discriminator network')
    parser.add_argument('--d_lr',type=float,default=1e-4,
                        help='learning rate for Discriminator optimizer')
    parser.add_argument('--d_mom',type=float,default=0.8,
                        help='momentum for Discriminator optimizer (ignored for Adam)')
    parser.add_argument('--g_opt',type=str,default='adam',help='optimizer for Generator network')
    parser.add_argument('--g_lr',type=float,default=1e-3,
                        help='learning rate for Generator optimizer')
    parser.add_argument('--g_mom',type=float,default=0.9,
                        help='momentum for Generator optimizer (ignored for Adam)')
    parser.add_argument('--sigma',type=float,default=0.15,
                        help='standard deviation of noise distribution to sample from')
   
    parser.add_argument('--cmap',type=str,default=None,help='matplotlib color map')
    
    parser.add_argument('--normalize',type=bool,default=True,help='normalize images or not')
    parser.add_argument('--explore',type=bool,default=True,
                        help='explore or debug after model runs')
    args = parser.parse_args()

    
    gan = mk_custom_gan(num_conv_layers=args.num_conv_layers,
                        nc=args.channels,
                        noise_dim=args.noise_dim)
    
    gan_results=train_gan(path=args.path,
                      model=gan, 
                      noise_dim=args.noise_dim, 
                      batch_size=args.batch_size, 
                      epochs=args.epochs, 
                      transform='default',
                      img_size=(args.img_size,args.img_size),
                      opt_g=args.g_opt,
                      opt_d=args.d_opt, 
                      lr_g=args.g_lr, 
                      lr_d=args.d_lr, 
                      weight_decay_g=0, 
                      weight_decay_d=0, 
                      mom_g=args.g_mom, 
                      mom_d=args.d_mom, 
                      sigma=args.sigma, 
                      loss_function='bce', 
                      cmap=args.cmap, 
                      normalize_plot=args.normalize,
                      plot_batches=False)
    
    if args.explore == True:
        import pdb
        pdb.set_trace()