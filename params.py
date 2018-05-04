import torch
import torch.nn as nn
import argparse

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=30,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    p.add_argument('-max_length', type=int, default=10,
                   help='maximum sequence length')
    p.add_argument('-output', type=str, default='output',
                   help='output directory for model saving')
    p.add_argument('-hidden', type=int, default=256,
                   help='dimension of hidden states')
    p.add_argument('-stack_size', type=int, default=10,
                   help='stack size')
    p.add_argument('-stack_elem_size', type=int, default=256,
                   help='dimension of each stack element')
    p.add_argument('-nstack', type=int, default=2,
                   help='how many stacks to use')
    p.add_argument('-stack_depth', type=int, default=2,
                   help='how many stack element to use for predicting')
    p.add_argument('-gpu',type=int,default=0,
                   help='gpu index(if could be used)')
    p.add_argument('-use_stack',type=bool, default=False,
                   help='whether to use stack')
    p.add_argument('-teaching', type=float, default=0.5,
                   help='teacher forcing ratio')
    return p.parse_args()

args=parse_arguments()
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

SOS=0
EOS=1
PAD=2
NACT=3
NONLINEAR=nn.Tanh
PUSH=0
POP=1
NOOP=2
