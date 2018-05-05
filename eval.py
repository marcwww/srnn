from main import to_batch
from main import pairs
from main import trans_one_sen

def train_accuracy(enc,dec):
    num=0
    for src,tar in pairs:
        res=trans_one_sen(enc,dec,src)
        num+=int(res==tar)

    return num/len(pairs)


