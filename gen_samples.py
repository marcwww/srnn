import random
def task1_one_seq(nmax,nmin,nchar):
    n=int((random.random() * (nmax-nmin))+nmin)
    p=['a']*nchar*n
    for c in range(nchar):
        for i in range(c*n,(c+1)*n):
            p[i]=chr(ord('a')+c)
    return p

#a^nb^{kn}
def task2_one_seq(nmax,nmin,nchar,k):
    n=int((random.random() * (nmax-nmin))+nmin)
    c=int(random.random()*(nchar-1)+1)
    p=['a']*(k+1)*n
    for i in range(n,len(p)):
        p[i]=chr(ord('a')+c)

    print(n,p)
    return p

def sample(max_length,nmax,nmin,nchar):
    res=[]
    while(True):
        seq=task1_one_seq(nmax,nmin,nchar)
        # seq = task2_one_seq(nmax, nmin, nchar,2)
        if len(res)+len(seq)>max_length:
            break
        res.extend(seq)
    return res

def samples(max_length,nseq,nmax,nmin,nchar):
    num=int(nseq/max_length)
    res=[]
    for _ in range(num):
        res.append(sample(max_length,nmax,nmin,nchar))

    return res

def gen_train(max_length=50,epoch=100,nseq=2000,nmin=2,nmax=5,nchar=5):
    res=samples(max_length, epoch*nseq, nmin, nmax, nchar)

    with open('corpus_train.txt','w') as f:
        for sample in res:
            f.write(' '.join(sample))
            f.write('\n')

def gen_test(max_length=50,nmaxmax=60,ntest=200,nchar=3):
    res=[]
    for nm in range(2,nmaxmax):
        nmin=nm
        nmax=nm+1
        res.extend(samples(max_length,ntest,nmax,nmin,nchar))

    with open('corpus_test.txt', 'w') as f:
        for sample in res:
            f.write(' '.join(sample))
            f.write('\n')

gen_train()
gen_test()