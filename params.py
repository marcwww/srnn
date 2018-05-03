import torch

SOS=1
EOS=0
MAX_LENGTH=50
use_cuda = torch.cuda.is_available()
BATCH_SIZE=512
LR=0.1
NEPOCHS=30
NACT=3
PUSH=0
POP=1
NOOP=2
EMPTY_VAL=-1
OUTPUT='output'
