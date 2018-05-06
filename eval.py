from main import to_batch
from main import pairs
from main import trans_one_batch
from main import batch_pairs
from main import output_lang
import torch
from params import args
import numpy as np
from params import EOS
BATCH_SIZE=args.batch_size

def train_accuracy(enc,dec):

    num=0
    total=0

    def cmp(pre,tar):
        pre=list(pre)
        tar=list(tar)
        print(pre[:len(tar)],tar)
        for i in range(len(tar)):
            if tar[i]!=pre[i]:
                return False

            if tar[i]==EOS:
                return True

    for i in range(len(batch_pairs)):
        src = batch_pairs[i][0]
        tar = batch_pairs[i][1]
        dec_tar = tar[1:, :]
        res, outputs = trans_one_batch(enc,dec,src)
        for i in range(BATCH_SIZE):
            if cmp(np.array((res[i])),dec_tar[:,i].cpu().numpy()):
               num+=1

        total+=BATCH_SIZE

    return num/total


