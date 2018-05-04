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
from torch.nn.utils import clip_grad_norm

SOS=params.SOS
EOS=params.EOS
PAD=params.PAD
GRAD_CLIP=params.GRAD_CLIP
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

    for i in range(len(pairs)):
        indices_src=indexesFromSentence(input_lang,pairs[i][0])
        indices_tar=indexesFromSentence(output_lang,pairs[i][1])
        batch_src.append(indices_src)
        batch_tar.append(indices_tar)

        if (i+1) % batch_size == 0:
            padded_src=[F.pad(torch.LongTensor(sen+[EOS]),(PAD,max_length+1-len(sen)))
                        for sen in batch_src]
            padded_tar=[F.pad(torch.LongTensor([SOS]+sen+[EOS]),(PAD,1+max_length+1-len(sen)))
                        for sen in batch_tar]

            # the transposing makes the data of the size: length * batch_size
            res.append((
                        torch.stack(padded_src).t().contiguous().to(DEVICE),
                        torch.stack(padded_tar).t().contiguous().to(DEVICE)
                        )
                       )
            batch_src=[]
            batch_tar=[]

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

def train(enc_optim,dec_optim,epoch,print_per_percent=0.1):

    total_loss=0
    pre_loss=total_loss
    t=time.time()

    batch_pairs_shuffle=to_batch(input_lang,output_lang,pairs,
                     batch_size=BATCH_SIZE,max_length=MAX_LENGTH)

    print_every=int(len(batch_pairs)*print_per_percent)
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
        output_indices=[]
        for dec_input in dec_inputs:
            # dec_input: shape of [batch_size]
            output, hidden, output_index = dec(dec_input,hidden,stacks)
            outputs.append(output)
            output_indices.append(output_index)

        # print(' '.join([output_lang.index2word[index[0].item()] for index in output_indices]))
        # print(' '.join([output_lang.index2word[index.item()] for index in dec_tar[:,0]]))
        outputs=torch.stack(outputs).view(-1,output_lang.n_words)
        loss=F.cross_entropy(outputs,dec_tar.view(-1),ignore_index=PAD)
        clip_grad_norm(enc.parameters(), max_norm=GRAD_CLIP)
        clip_grad_norm(dec.parameters(),max_norm=GRAD_CLIP)

        loss.backward()
        enc_optim.step()
        dec_optim.step()
        total_loss+=loss.item()

        # pair = random.choice(pairs)
        # print('src:',pair[0],'tar_pred:',trans_one_sen(pair[0]),'tar_ground:',pair[1])

        if (i+1) % print_every == 0:
            total_loss=total_loss/print_every
            print('epoch %d | percent %f | loss %f | interval %f s' %
                  (epoch,
                   i/len(batch_pairs),
                   total_loss,
                   time.time()-t))
            t=time.time()
            pre_loss = total_loss
            total_loss=0

            eval_randomly(n=1)

    return pre_loss

def trans_one_sen(src,max_length=MAX_LENGTH):
    with torch.no_grad():
        indices=indexesFromSentence(input_lang,src)
        # src_batch: length * (batch_size=1)
        # src_batch=torch.LongTensor(indices+[EOS]).unsqueeze(0).t().to(DEVICE)
        padded_src = F.pad(torch.LongTensor(indices + [EOS]), (PAD, max_length + 1 - len(indices)))
        padded_src=padded_src.unsqueeze(0).t().to(DEVICE)

        hidden = enc.init_hidden(batch_size=1)
        stacks = enc.init_stack(batch_size=1)

        _, hidden, stacks = enc(padded_src, hidden, stacks)
        dec_input = torch.LongTensor([SOS]).to(DEVICE)

        output_indices=[]
        while len(output_indices)<max_length:
            # dec_input: shape of [batch_size=1]
            _, hidden, output_index = dec(dec_input, hidden, stacks)
            if output_index.item()==EOS:
                break
            output_indices.append(output_index.item())
            dec_input = output_index.squeeze(0)

        return ' '.join([output_lang.index2word[output_index] for output_index in output_indices])

def eval_randomly(n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = trans_one_sen(pair[0])
        print('<', output_words)
        print('')

def train_epochs():
    criterion = nn.NLLLoss()
    # enc_optim=optim.Adagrad(enc.parameters(),lr=LR)
    # dec_optim=optim.Adagrad(dec.parameters(),lr=LR)
    enc_optim = optim.SGD(enc.parameters(), lr=LR)
    dec_optim = optim.SGD(dec.parameters(), lr=LR)
    best_loss = None
    name = ''.join(str(time.time()).split('.'))
    enc_file = OUTPUT + '/' + 'enc_' + name + '.pt'
    dec_file = OUTPUT + '/' + 'dec_' + name + '.pt'

    for epoch in range(NEPOCHS):
        epoch_start_time = time.time()
        loss = train(enc_optim, dec_optim, epoch)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            with open(enc_file, 'wb') as f:
                torch.save(enc, f)
            with open(dec_file, 'wb') as f:
                torch.save(dec, f)

        print('end of epoch %d | time: %f s | loss: %f' %
              (epoch,
               time.time() - epoch_start_time,
               loss))

if __name__ == '__main__':
    # name='15254220464367697'
    # enc_file = OUTPUT + '/' + 'enc_' + name + '.pt'
    # dec_file = OUTPUT + '/' + 'dec_' + name + '.pt'
    # with open(enc_file, 'rb') as f:
    #     enc=torch.load(f,map_location='cpu')
    # with open(dec_file, 'rb') as f:
    #     dec=torch.load(f,map_location='cpu')

    train_epochs()
    # eval_randomly(100)


