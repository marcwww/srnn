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

def samples(nseq,nmax,nmin,nchar):
    res=[]
    for _ in range(nseq):
        res.append(list(task1_one_seq(nmax,nmin,nchar)))
    return res

def gen_train(nseq=2000*100,nmin=2,nmax=20,nchar=3):
    res=samples(nseq, nmin, nmax, nchar)

    with open('corpus_train.txt','w') as f:
        for sample in res:
            f.write(' '.join(sample))
            f.write('\n')

def gen_test(nmaxmax=60,ntest=1,nchar=3):
    res=[]
    for nm in range(2,nmaxmax):
        nmin=nm
        nmax=nm+1
        res.extend(samples(ntest,nmax,nmin,nchar))

    with open('corpus_test.txt', 'w') as f:
        for sample in res:
            f.write(' '.join(sample))
            f.write('\n')

gen_train()
gen_test()