import data
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import random
import stack
from torch import nn
from torch import optim
import time
import crash_on_ipy
import params

SOS=params.SOS
EOS=params.EOS
MAX_LENGTH=params.MAX_LENGTH
use_cuda = params.use_cuda
BATCH_SIZE=params.BATCH_SIZE
LR=params.LR
NEPOCHS=params.NEPOCHS

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def to_batch(input_lang, output_lang,
             pairs, batch_size, max_length):
    res=[]
    batch_src=[]
    batch_tar=[]
    pairs=list(random.sample(pairs,len(pairs)))

    max_length_src=0
    max_length_tar=0

    for i in range(len(pairs)):
        indices_src=indexesFromSentence(input_lang,pairs[i][0])
        indices_tar=indexesFromSentence(output_lang,pairs[i][1])
        batch_src.append(indices_src)
        batch_tar.append(indices_tar)
        max_length_src=max(max_length_src,len(indices_src))
        max_length_tar=max(max_length_tar,len(indices_tar))
        if (i+1) % batch_size == 0:
            max_length_src=min(max_length_src,MAX_LENGTH)
            max_length_tar=min(max_length_tar,MAX_LENGTH)

            padded_src=[F.pad(torch.LongTensor(sen+[EOS]),(0,max_length_src+1-len(sen)))
                        for sen in batch_src]
            padded_tar=[F.pad(torch.LongTensor([SOS]+sen+[EOS]),(0,1+max_length_tar+1-len(sen)))
                        for sen in batch_tar]

            res.append((
                        torch.stack(padded_src).t().contiguous().cuda()
                        if use_cuda else torch.stack(padded_src).t().contiguous(),

                        torch.stack(padded_tar).t().contiguous().cuda()
                        if use_cuda else torch.stack(padded_tar).t().contiguous()
                        )
                       )
            batch_src=[]
            batch_tar=[]
            max_length_src=0
            max_length_tar=0

    # res: list of batch pairs
    return res

input_lang, output_lang, pairs = data.prepareData('spa', 'en', True)
batch_pairs=to_batch(input_lang,output_lang,pairs,
                     batch_size=BATCH_SIZE,max_length=MAX_LENGTH)

enc=stack.EncoderSRNN(input_lang.n_words,
                  hidden_size=256,
                  nstack=2,
                  stack_depth=2,
                  stack_size=10,
                  stack_elem_size=256,
                  batch_size=BATCH_SIZE)
dec=stack.DecoderSRNN(hidden_size=256,
                  output_size=output_lang.n_words,
                  nstack=2,
                  stack_depth=2,
                  stack_size=10,
                  stack_elem_size=256,
                  batch_size=BATCH_SIZE)
if use_cuda:
    enc.cuda()
    dec.cuda()

def train(enc_optim,dec_optim,criterion,epoch,print_per_percent=0.1):

    total_loss=0
    t=time.time()
    print_every=int(len(batch_pairs)*print_per_percent)

    for i in range(len(batch_pairs)):
        src=batch_pairs[i][0]
        tar=batch_pairs[i][1]
        hidden=enc.init_hidden(BATCH_SIZE)
        stacks=enc.init_stack(BATCH_SIZE)

        dec_inputs=tar[:-1,:]
        dec_tar=tar[1:,:]
        _, hidden, stacks = enc(src,hidden,stacks,batch_size=BATCH_SIZE)

        outputs, _, _ = dec(dec_inputs,hidden,stacks,
                             batch_size=BATCH_SIZE)

        loss=0.0
        for oi in range(len(outputs)):
            loss+=criterion(outputs[oi],dec_tar[oi])

        loss.backward()
        enc_optim.step()
        dec_optim.step()
        total_loss+=loss.data

        if (i+1) % print_every == 0:
            print('epoch %d | percent %f | loss %f | interval %f s' %
                  (epoch,
                   i/len(batch_pairs),
                   total_loss/(i*BATCH_SIZE),
                   time.time()-t))
            t=time.time()
            pair=random.choice(pairs)
            print(eval(pair[0]),pair[1])

    return total_loss/(len(batch_pairs)*BATCH_SIZE+.0)

def eval(src):
    indices=indexesFromSentence(input_lang,src)
    res=torch.LongTensor(indices).expand(BATCH_SIZE,len(indices))
    res=res.t()

    hidden = enc.init_hidden(BATCH_SIZE)
    stacks = enc.init_stack(BATCH_SIZE)

    dec_inputs=torch.LongTensor([SOS]*MAX_LENGTH).expand(BATCH_SIZE,MAX_LENGTH)
    dec_inputs=dec_inputs.t()

    _, hidden, stacks = enc(res, hidden, stacks, batch_size=BATCH_SIZE)

    outputs, _, indices = dec(dec_inputs, hidden, stacks,
                        batch_size=BATCH_SIZE)

    indices=indices[0]
    return ' '.join([output_lang.index2word[index] for index in indices])

if __name__ == '__main__':
    criterion=nn.NLLLoss()
    enc_optim=optim.Adagrad(enc.parameters(),lr=LR)
    dec_optim=optim.Adagrad(dec.parameters(),lr=LR)
    for epoch in range(NEPOCHS):
        epoch_start_time=time.time()
        loss=train(enc_optim,dec_optim,criterion,epoch)

        print('end of epoch %d | time: %f s | loss: %f' %
              (epoch,
               time.time()-epoch_start_time,
               loss))

