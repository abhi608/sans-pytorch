from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import pdb
import h5py
import json
import re
import math
import torch


def main(params):
    torch.manual_seed(params['seed'])
    torch.set_default_tensor_type('torch.FloatTensor')
    

    # pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_img_train_h5', default='data/vqa_data_img_vgg_train.h5', help='path to the h5file containing the train image feature')
    parser.add_argument('--input_img_test_h5', default='data/vqa_data_img_vgg_test.h5', help='path to the h5file containing the test image feature')
    parser.add_argument('--input_ques_h5', default='data/vqa_data_prepro.h5', help='path to the json file containing additional info and vocab')

    parser.add_argument('--input_json', default='data/vqa_data_prepro.json', help='output json file')
    parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = don\'t')
  
    # Options
    parser.add_argument('--feature_type', default='VGG', help='VGG or Residual')
    parser.add_argument('--emb_size', default=500, type=int, help='the size after embeeding from onehot')
    parser.add_argument('--hidden_size', default=1024, type=int, help='the hidden layer size of the model')
    parser.add_argument('--rnn_size', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--att_size', default=512, type=int, help='size of sttention vector which refer to k in paper')
    parser.add_argument('--batch_size', default=200, type=int, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--output_size', default=1000, type=int, help='number of output answers')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of the rnn layer')


    # Optimization
    parser.add_argument('--optim', default='rmsprop', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', default=4e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--learning_rate_decay_start', default=100, type=int, help='at what iteration to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', default=1500, type=int, help='every how many epoch thereafter to drop LR by 0.1?')
    parser.add_argument('--optim_alpha', default=0.99, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--optim_beta', default=0.995, type=float, help='beta used for adam')
    parser.add_argument('--optim_epsilon', default=1e-8, type=float, help='epsilon that goes into denominator in rmsprop')
    parser.add_argument('--max_iters', default=-1, type=int, help='max number of iterations to run for (-1 = run forever)')
    parser.add_argument('--iterPerEpoch', default=1250, type=int, help=' no. of iterations per epoch')

    # Evaluation/Checkpointing
    parser.add_argument('--save_checkpoint_every', default=6000, type=int, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', default='save/train_vgg', help='folder to save checkpoints into (empty = this folder)')

    # Visualization
    parser.add_argument('--losses_log_every', default=600, type=int, help='How often do we save losses, for inclusion in the progress dump? (0 = disable)')

    # misc
    parser.add_argument('--id', default='1', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
    parser.add_argument('--gpuid', default=2, type=int, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--seed', default=1234, type=int, help='random number generator seed to use')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)