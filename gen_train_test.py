import random

def task1_one_seq(nmax,nmin,nchar):
    n=int((random.random() * (nmax-nmin))+nmin)
    p=['a']*nchar*n
    for c in range(nchar):
        for i in range(c*n,(c+1)*n):
            p[i]=chr(ord('a')+c)
    return p

def gen(nepoch=100,nseq=2000,bptt=50,nmaxmax=10,nmin=2,nchar=2):
    with open('corpus_train.txt','w') as f:

        for e in range(nepoch):
            longseq=''
            nmax = max(min(e + 3, nmaxmax), 3)
            for iseq in range(nseq):
                p=task1_one_seq(nmax,nmin,nchar)
                longseq+=''.join(p)

            for i in range(len(longseq)-bptt):
                to_write=longseq[i:i+bptt]
                f.write(' '.join(list(to_write)))
                f.write('\n')

gen()








