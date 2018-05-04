import torch

SOS=1
EOS=0
MAX_LENGTH=10
use_cuda = torch.cuda.is_available()
# BATCH_SIZE=512
BATCH_SIZE=5
LR=0.01
NEPOCHS=30
NACT=3
PUSH=0
POP=1
NOOP=2
EMPTY_VAL=-1
OUTPUT='output'
USES_STACK=False
