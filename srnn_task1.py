import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import random
import math
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import dot_graph

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 1000
HIDDEN_SIZE=10
EMPTY_VAL=-1
NACT=3 #pop,push and noop
PUSH=0
POP=1
NOOP=2
STACK_ELEM_SIZE=1
NCHAR=3

class WordCollection:
    def __init__(self,name):
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word=[]
        self.n_words=len(self.index2word)

    def add_sen(self,sen):
        for word in sen.strip().split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word]=1
            self.index2word.append(word)
            self.n_words+=1
        else:
            self.word2count[word]+=1

def shift_matrix(n):
    W_up=np.eye(n)
    for i in range(n-1):
        W_up[i,:]=W_up[i+1,:]
    W_up[n-1,:]*=0
    W_down=np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:]=W_down[i-1,:]
    W_down[0,:]*=0
    return W_up,W_down

def create_stack(size):
    return np.array((([EMPTY_VAL]*STACK_ELEM_SIZE))*size)

class SRNNPredictor(nn.Module):

    # voc_size = size of input vocabulary = size of output vocabulary
    def __init__(self,voc_size,input_size,hidden_size,
                 nstack,stack_depth,stack_size):
        super(SRNNPredictor,self).__init__()
        self.hidden_size=hidden_size
        self.nstack=nstack
        self.stack_size=stack_size
        self.stack_depth=stack_depth
        self.embedding=nn.Embedding(voc_size,input_size)
        self.nonLinear=nn.Tanh()
        # self.nonLinear = nn.Sigmoid()
        self.hid2hid=nn.Linear(hidden_size,hidden_size)
        self.input2hid=nn.Linear(input_size,hidden_size)
        self.log_softmax=nn.LogSoftmax(dim=1)
        self.softmax=nn.Softmax()
        self.hid2out=nn.Linear(hidden_size,voc_size)
        self.hid2act=[nn.Linear(hidden_size,NACT)
                      for _ in range(nstack)]
        self.hid2stack=[nn.Linear(hidden_size,STACK_ELEM_SIZE)
                        for _ in range(nstack)]
        self.stack2hid=[nn.Linear(STACK_ELEM_SIZE*stack_depth,hidden_size)
                        for _ in range(nstack)]
        empty_stack=create_stack(stack_size)
        empty_stack=torch.Tensor(empty_stack)
        self.stacks=[Variable(empty_stack)]*nstack

        self.bias=Variable(torch.zeros(1,hidden_size),requires_grad=True)
        W_up,W_down=shift_matrix(stack_size)
        self.W_up=Variable(torch.Tensor(W_up))
        self.W_down=Variable(torch.Tensor(W_down))

    def forward(self, input, hidden):
        emb=self.embedding(input).view(1,1,-1)
        middle_hidden = self.input2hid(emb) + self.hid2hid(hidden)
        for stack_index in range(self.nstack):
            # Note: it is the **previous** stack info given to the hidden at now time.
            stack_vals = self.stacks[stack_index][0:self.stack_depth].view(-1)
            middle_hidden += self.stack2hid[stack_index](stack_vals)

            act=self.hid2act[stack_index](hidden).view(-1)
            act=self.softmax(act)

            push_val=self.hid2stack[stack_index](hidden)
            push_val=self.softmax(push_val)

            self.stacks[stack_index]= \
                act[PUSH]*self.W_down.matmul(self.stacks[stack_index])+ \
                act[POP]*self.W_up.matmul(self.stacks[stack_index])+ \
                act[NOOP]*self.stacks[stack_index]
            self.stacks[stack_index][0]=act[PUSH]*push_val
            self.stacks[stack_index][self.stack_size-1]=act[POP]*EMPTY_VAL

        middle_hidden=self.nonLinear(middle_hidden)
        output=self.hid2out(middle_hidden)[0]
        output=self.log_softmax(output)
        hidden=middle_hidden
        return output,hidden

    def emtpy_stack(self):
        empty_stack = create_stack(self.stack_size)
        empty_stack = torch.Tensor(empty_stack)
        self.stacks = [Variable(empty_stack)] * self.nstack

    def init_hidden(self):
        res=Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return res.cuda()
        else:
            return res

def train(hidden,cur,next,predictor,optimizer,criterion,back_update,idx):
    optimizer.zero_grad()
    output,hidden=predictor(cur,hidden)
    loss=criterion(output,next)
    # g=dot_graph.make_dot(hidden)
    # g.view(str(idx))

    if back_update:
        loss.backward(retain_graph=True)
        # loss.backward()
        optimizer.step()

    return hidden,loss

def test(corpus,target_var,predictor):
    ncorr=0
    nword=0
    tar=[corpus.index2word[target_var[0].data[0]]]
    pred=[corpus.index2word[target_var[0].data[0]]]
    target_length=target_var.size()[0]
    hidden=predictor.init_hidden()
    iseval=False
    for si in range(target_length-1):
        output, hidden = predictor(target_var[si], hidden)
        topv, topi = output.data.topk(1)
        ni = topi[0][0]
        pred.append(corpus.index2word[ni])
        tar.append(corpus.index2word[target_var[si+1].data[0]])

        if iseval:
            nword+=1
            if ni==target_var[si+1].data[0]:
                ncorr+=1
        if target_var[si].data[0]!=0 and \
                target_var[si+1].data[0]==0:
            pred.append('_-')
            tar.append('_-')
            iseval=False
        if target_var[si].data[0]==0 and \
                target_var[si+1].data[0]!=0:
            pred.append('-_')
            tar.append('-_')
            iseval=True

    return ncorr,nword,pred,tar

def normalize_string(str):
    return str

def read_corpus(corpus):
    print('Reading lines...')
    lines=open('data/%s.txt' % corpus,encoding='utf-8').read().strip().split('\n')
    lines=[normalize_string(line) for line in lines]
    input_corpus=WordCollection(corpus)
    return input_corpus,lines

def filter_line(line):
    return len(line.split(' '))<MAX_LENGTH

def filter_lines(lines):
    return [line for line in lines if filter_line(line)]

def prepare_data(coprus):
   input_corpus,lines=read_corpus(coprus)
   print('Read %d sentences' % len(lines))
   lines=filter_lines(lines)
   print('Trimmed to %d sentences' % len(lines))
   print('Counting words...')
   for line in lines:
       input_corpus.add_sen(line)
   print('Counted words:')
   print(input_corpus.name,input_corpus.n_words)
   return input_corpus,lines

def indices_from_line(corpus,line):
    return [corpus.word2index[word] for word in line.split(' ')]

def var_from_line(corpus,line):
    indices=indices_from_line(corpus,line)
    res=Variable(torch.LongTensor(indices).view(-1,1))
    if use_cuda:
        return res.cuda()
    else:
        return res

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def task1_one_seq(nmax,nmin,nchar):
    n=int((random.random() * (nmax-nmin))+nmin)
    p=['a']*nchar*n
    for c in range(nchar):
        for i in range(c*n,(c+1)*n):
            p[i]=chr(ord('a')+c)
    return p

def indices_from_seq(corpus,seq):
    return [corpus.word2index[word] for word in list(seq)]

def vars_from_seq(corpus,seq):
    indices = indices_from_seq(corpus, seq)
    res = Variable(torch.LongTensor(indices).view(-1, 1))
    if use_cuda:
        return res.cuda()
    else:
        return res

def gen_test(nmaxmax=60,nchar=NCHAR):
    res=[]
    for nm in range(2,nmaxmax):
        nmin=nm
        nmax=nm+1
        res.append(task1_one_seq(nmax,nmin,nchar))

    return res

def train_iters(predictor,corpus,n_iters=100,print_every=1000,
                plot_every=100,learning_rate=1e-1,bptt=50):
    start=time.time()
    plot_losses=[]
    print_loss_total=0
    plot_loss_total=0
    nseq=2000
    nmax=20
    nmin=2
    hidden = predictor.init_hidden()
    retain_graph=True
    bptt_counter=0
    nreset=1000
    back_update=True
    predictor.emtpy_stack()
    losses=[0]*bptt

    # optimizer=optim.SGD(predictor.parameters(),lr=learning_rate)
    optimizer=optim.Adagrad(predictor.parameters(),lr=learning_rate)
    criterion=nn.NLLLoss()
    for iter in range(1,n_iters+1):
        for iseq in range(nseq):
            seq=task1_one_seq(nmax,nmin,NCHAR)
            vars=vars_from_seq(corpus,seq)
            if iseq % nreset == 0:
                predictor.emtpy_stack()

            for i in range(len(vars)-1):
                if i == 0 and iseq == 0:
                    predictor.emtpy_stack()
                    back_update=False
                else:
                    back_update=True

                print('a')
                hidden,loss=\
                    train(hidden,vars[i],vars[i+1],
                          predictor,optimizer,criterion,back_update,bptt_counter)
                loss=loss.detach()
                # hidden=predictor.init_hidden()
                # losses[bptt_counter % bptt]=loss
                #
                # if bptt_counter % bptt == bptt-1:
                #     losses[0].detach_()
                #     print('a')

                bptt_counter += 1


                print_loss_total+=loss.data[0]
                plot_loss_total+=loss.data[0]

            # print('a')



            if iter % print_every == 0:
                print_loss_avg=print_loss_total/print_every
                print_loss_total=0
                print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
                eval(predictor, corpus)
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    show_plot(plot_losses)

def eval(predictor,corpus):
    ncorr=.0
    ntotal=0
    seqs=gen_test()
    with open('data/res.txt','w') as f:
        for seq in seqs:
            vars=vars_from_seq(corpus,seq)
            nc,nt,pred,tar=test(corpus,vars,predictor)
            f.write('-tar:%s,\npred:%s, correct:%d\n' % (''.join(tar),''.join(pred),int(nc==nt)))
            ncorr+=nc
            ntotal+=nt
    print(ncorr/ntotal)

if __name__=='__main__':
    corpus=WordCollection('corpus')
    for i in range(NCHAR):
        corpus.add_word(chr(ord('a')+i))

    predictor=SRNNPredictor(voc_size=corpus.n_words,
                           input_size=HIDDEN_SIZE,
                           hidden_size=HIDDEN_SIZE,nstack=1,stack_depth=2,stack_size=200)
    if use_cuda:
        predictor=predictor.cuda()

    train_iters(predictor,corpus,100,print_every=1,plot_every=1)