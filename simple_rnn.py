import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import random
import math
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 100
EOS_index=0
HIDDEN_SIZE=50

class WordCollection:
    def __init__(self,name):
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word=['']
        self.index2word[EOS_index]='EOS'
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

class SeqPredictor(nn.Module):

    # voc_size = size of input vocabulary = size of output vocabulary
    def __init__(self,voc_size,input_size,hidden_size):
        super(SeqPredictor,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(voc_size,input_size)
        # self.rnn=nn.RNNCell(input_size,hidden_size)
        self.gru=nn.GRU(input_size,hidden_size)
        self.softmax=nn.LogSoftmax(dim=1)
        self.out=nn.Linear(hidden_size,voc_size)

    def forward(self, input, hidden):
        emb=self.embedding(input).view(1,1,-1)
        # hidden=self.rnn(emb,hidden)
        # output=hidden
        output,hidden=self.gru(emb,hidden)
        output=self.out(output)[0]
        output=self.softmax(output)
        return output,hidden

    def init_hidden(self):
        res=Variable(torch.zeros(1,1,self.hidden_size))
        if use_cuda:
            return res.cuda()
        else:
            return res

def train(target_var,predictor,optimizer,criterion,max_length=MAX_LENGTH):
    loss=0
    target_length=target_var.size()[0]
    hidden=predictor.init_hidden()
    optimizer.zero_grad()

    # line=[]
    for si in range(target_length-1):
        output,hidden=predictor(target_var[si],hidden)
        # topv, topi = output.data.topk(1)
        # ni = topi[0][0]
        # line.append(ni)
        # print(ni,target_var[si+1])
        loss+=criterion(output,target_var[si+1])
    # print(line)

    loss.backward()
    nn.utils.clip_grad_norm(predictor.parameters(),5)
    optimizer.step()

    return loss.data[0]/target_length

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
    indices.append(EOS_index)
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

def train_iters(predictor,corpus,lines,n_iters,print_every=1000,plot_every=100,learning_rate=1e-1):
    start=time.time()
    plot_losses=[]
    print_loss_total=0
    plot_loss_total=0

    # optimizer=optim.SGD(predictor.parameters(),lr=learning_rate)
    optimizer=optim.Adagrad(predictor.parameters(),lr=learning_rate)
    training_inputs=[var_from_line(corpus,random.choice(lines)) for _ in range(n_iters)]
    criterion=nn.NLLLoss()
    for iter in range(1,n_iters+1):
        target_var=training_inputs[iter-1]
        loss=train(target_var,predictor,optimizer,criterion)
        print_loss_total+=loss
        plot_loss_total+=loss

        if iter % print_every == 0:
            print_loss_avg=print_loss_total/print_every
            print_loss_total=0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    show_plot(plot_losses)

if __name__=='__main__':
    input_corpus,lines=prepare_data('corpus')

    predictor=SeqPredictor(voc_size=input_corpus.n_words,
                           input_size=HIDDEN_SIZE,
                           hidden_size=HIDDEN_SIZE)
    if use_cuda:
        predictor=predictor.cuda()

    train_iters(predictor,input_corpus,lines,10000,print_every=200,plot_every=200)