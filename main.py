import data
import torch.nn.functional as F
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
OUTPUT=params.OUTPUT
USE_STACK=params.USES_STACK
DEVICE=params.device

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
            max_length_src=min(max_length_src,max_length)
            max_length_tar=min(max_length_tar,max_length)

            padded_src=[F.pad(torch.LongTensor(sen+[EOS]),(0,max_length_src+1-len(sen)))
                        for sen in batch_src]
            padded_tar=[F.pad(torch.LongTensor([SOS]+sen+[EOS]),(0,1+max_length_tar+1-len(sen)))
                        for sen in batch_tar]

            # the transposing makes the data of the size: length * batch_size
            res.append((
                        torch.stack(padded_src).t().contiguous().to(DEVICE),
                        torch.stack(padded_tar).t().contiguous().to(DEVICE)
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
                  stack_elem_size=256).to(DEVICE)
dec=stack.DecoderSRNN(hidden_size=256,
                  output_size=output_lang.n_words,
                  nstack=2,
                  stack_depth=2,
                  stack_size=10,
                  stack_elem_size=256).to(DEVICE)

def train(enc_optim,dec_optim,criterion,epoch,print_per_percent=0.1):

    total_loss=0
    pre_loss=total_loss
    t=time.time()
    print_every=int(len(batch_pairs)*print_per_percent)

    batch_pairs_shuffle=random.sample(batch_pairs,k=len(batch_pairs))
    for i in range(len(batch_pairs_shuffle)):
        # one source batch and one target batch:
        # src: length * batch_size
        # tar: length * batch_size
        src=batch_pairs_shuffle[i][0]
        tar=batch_pairs_shuffle[i][1]
        hidden=enc.init_hidden(BATCH_SIZE)
        stacks=enc.init_stack(BATCH_SIZE)

        # dec_inputs start with [BOS]
        dec_inputs=tar[:-1,:]
        # dec_outputs end with [EOS]
        dec_tar=tar[1:,:]
        _, hidden, stacks = enc(src,hidden,stacks)

        outputs=[]
        for dec_input in dec_inputs:
            # dec_input: shape of [batch_size]
            output, hidden, output_index = dec(dec_input,hidden,stacks)
            outputs.append(output)

        loss=0.0
        for oi in range(len(outputs)):
            loss+=criterion(outputs[oi],dec_tar[oi])

        loss.backward()
        enc_optim.step()
        dec_optim.step()
        total_loss+=loss.data/len(outputs)

        # pair = random.choice(pairs)
        # print('src:',pair[0],'tar_pred:',trans_one_sen(pair[0]),'tar_ground:',pair[1])

        if (i+1) % print_every == 0:
            print('epoch %d | percent %f | loss %f | interval %f s' %
                  (epoch,
                   i/len(batch_pairs),
                   total_loss/print_every,
                   time.time()-t))
            t=time.time()
            pre_loss = total_loss / print_every
            total_loss=0

            pair = random.choice(pairs)
            print('src:',pair[0],'tar_pred:',trans_one_sen(pair[0]),'tar_ground:',pair[1])

    return pre_loss

def trans_one_sen(src,max_length=MAX_LENGTH):
    with torch.no_grad():
        indices=indexesFromSentence(input_lang,src)
        # src_batch: length * (batch_size=1)
        src_batch=torch.LongTensor(indices+[EOS]).unsqueeze(0).t().to(DEVICE)

        hidden = enc.init_hidden(batch_size=1)
        stacks = enc.init_stack(batch_size=1)

        dec_input = torch.LongTensor([SOS]).to(DEVICE)

        _, hidden, stacks = enc(src_batch, hidden, stacks)

        output_indices=[]
        while len(output_indices)<max_length:
            # dec_input: shape of [batch_size=1]
            _, hidden, output_index = dec(dec_input, hidden, stacks)
            if output_index.item()==EOS:
                break
            output_indices.append(output_index.item())
            dec_input = output_index.squeeze(0)

        return ' '.join([output_lang.index2word[output_index] for output_index in output_indices])

if __name__ == '__main__':
    criterion=nn.NLLLoss()
    # enc_optim=optim.Adagrad(enc.parameters(),lr=LR)
    # dec_optim=optim.Adagrad(dec.parameters(),lr=LR)
    enc_optim = optim.SGD(enc.parameters(), lr=LR)
    dec_optim = optim.SGD(dec.parameters(), lr=LR)
    best_loss=None
    name=''.join(str(time.time()).split('.'))
    enc_file=OUTPUT+'/'+'enc_'+name+'.pt'
    dec_file=OUTPUT+'/'+'dec_'+name+'.pt'

    for epoch in range(NEPOCHS):
        epoch_start_time=time.time()
        loss=train(enc_optim,dec_optim,criterion,epoch)
        if best_loss is None or loss<best_loss:
            best_loss=loss
            with open(enc_file,'wb') as f:
                torch.save(enc,f)
            with open(dec_file,'wb') as f:
                torch.save(dec,f)

        print('end of epoch %d | time: %f s | loss: %f' %
              (epoch,
               time.time()-epoch_start_time,
               loss))


