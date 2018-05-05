import data
import torch.nn.functional as F
import torch
import random
import stack
import stack_ex
from torch import nn
from torch import optim
import time
import crash_on_ipy
import params
from params import args
from params import device
import argparse
from torch.nn.utils import clip_grad_norm
import gru
import eval

SOS=params.SOS
EOS=params.EOS
PAD=params.PAD
GRAD_CLIP=args.grad_clip
MAX_LENGTH=args.max_length
BATCH_SIZE=args.batch_size
LR=args.lr
NEPOCHS=args.epochs
OUTPUT=args.output
USE_STACK=args.use_stack
DEVICE=device
TEACHING_RATIO=args.teaching

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def to_batch(input_lang, output_lang,
             pairs, batch_size):
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
            max_length_src = max([len(src) for src in batch_src])
            max_length_tar = max([len(tar) for tar in batch_tar])

            padded_src=[F.pad(torch.LongTensor(sen+[EOS]),
                              (0,max_length_src+1-len(sen+[EOS])),
                              value=PAD)
                        for sen in batch_src]
            padded_tar=[F.pad(torch.LongTensor([SOS]+sen+[EOS]),
                              (0,1+max_length_tar+1-len([SOS]+sen+[EOS])),
                              value=PAD)
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

# input_lang, output_lang, pairs = data.prepareData('spa', 'en', True)
input_lang, output_lang, pairs = data.prepareData('aa', 'bb', True)
batch_pairs=to_batch(input_lang,output_lang,pairs,
                     batch_size=BATCH_SIZE)

# enc = stack.EncoderSRNN(input_size=input_lang.n_words,
#                             hidden_size=args.hidden,
#                             nstack=args.nstack,
#                             stack_depth=args.stack_depth,
#                             stack_size=args.stack_size,
#                             stack_elem_size=args.stack_elem_size).\
#                             to(DEVICE)
# dec = stack.DecoderSRNN(output_size=output_lang.n_words,
#                             hidden_size=args.hidden,
#                             nstack=args.nstack,
#                             stack_depth=args.stack_depth,
#                             stack_size=args.stack_size,
#                             stack_elem_size=args.stack_elem_size)\
#                             .to(DEVICE)
enc = stack_ex.EncoderSRNN(input_size=input_lang.n_words,
                            hidden_size=args.hidden,
                            nstack=args.nstack,
                            stack_depth=args.stack_depth,
                            stack_size=args.stack_size,
                            stack_elem_size=args.stack_elem_size).\
                            to(DEVICE)
dec = stack_ex.DecoderSRNN(output_size=output_lang.n_words,
                            hidden_size=args.hidden,
                            nstack=args.nstack,
                            stack_depth=args.stack_depth,
                            stack_size=args.stack_size,
                            stack_elem_size=args.stack_elem_size)\
                            .to(DEVICE)
# enc = gru.Encoder(input_size=input_lang.n_words,
#                   hidden_size=args.hidden).to(DEVICE)
# dec = gru.Decoder(output_size=output_lang.n_words,
#                   hidden_size=args.hidden).to(DEVICE)

def no_teaching(hidden,stacks,dec,max_length=MAX_LENGTH):
    dec_input = torch.LongTensor([SOS]).expand(BATCH_SIZE)

    outputs = []
    output_indices = []
    batch_ends = [False] * BATCH_SIZE

    def ends(batch_ends):
        for end in batch_ends:
            if end == False:
                return False
        return True

    while len(output_indices) < max_length:
        # dec_input: shape of [batch_size]
        output, hidden, output_index, stacks = dec(dec_input, hidden, stacks)
        outputs.append(output)
        output_indices.append(output_index)
        for i in range(BATCH_SIZE):
            if output_index[i].item() == EOS:
                batch_ends[i] = False
        if ends(batch_ends):
            break

        dec_input = output_index.squeeze(1)

    return outputs, output_indices

def train(enc_optim,dec_optim,epoch,print_per_percent=0.1):

    total_loss=0
    pre_loss=total_loss
    t=time.time()

    # batch_pairs_shuffle=to_batch(input_lang,
    #                              output_lang,
    #                              pairs,
    #                              batch_size=BATCH_SIZE)
    #
    batch_pairs_shuffle=batch_pairs

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
        teaching=random.random() < args.teaching
        if teaching:
            for dec_input in dec_inputs:
                # dec_input: shape of [batch_size]
                output, hidden, output_index, stacks = dec(dec_input,hidden,stacks)
                outputs.append(output)
        else:
            outputs,_ =no_teaching(hidden,stacks,dec)
            outputs=outputs[:len(dec_tar)]

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
            with open(params.log_file,'a+') as f:
                print('epoch %d | percent %f | loss %f | interval %f s' %
                      (epoch,
                       i/len(batch_pairs),
                       total_loss,
                       time.time()-t),file=f)
            t=time.time()
            pre_loss = total_loss
            total_loss=0
            eval_randomly(n=1)

    return pre_loss

def trans_one_batch(enc,dec,src_batch,max_length=MAX_LENGTH):
    with torch.no_grad():
        hidden = enc.init_hidden(BATCH_SIZE)
        stacks = enc.init_stack(BATCH_SIZE)

        _, hidden, stacks = enc(src_batch, hidden, stacks)

        outputs, output_indices = no_teaching(hidden,stacks,dec)

        res=[]
        for i in range(BATCH_SIZE):
            one_batch=[]
            for j in range(len(output_indices)):
                idx=output_indices[j][i].item()
                # word=output_lang.index2word[idx]
                # res[i].append(word)
                one_batch.append(idx)
            res.append(one_batch)
        return res, outputs

def trans_one_sen(enc,dec,src,max_length=MAX_LENGTH):
    with torch.no_grad():
        indices=indexesFromSentence(input_lang,src)
        # src_batch: length * (batch_size=1)
        # src_batch=torch.LongTensor(indices+[EOS]).unsqueeze(0).t().to(DEVICE)
        # padded_src = F.pad(torch.LongTensor(indices + [EOS]),
        #                    (0, max_length + 1 - len(indices)),
        #                    value=PAD)
        padded_src = torch.LongTensor(indices + [EOS])
        padded_src=padded_src.unsqueeze(0).t().to(DEVICE)

        hidden = enc.init_hidden(batch_size=1)
        stacks = enc.init_stack(batch_size=1)

        _, hidden, stacks = enc(padded_src, hidden, stacks)
        dec_input = torch.LongTensor([SOS]).to(DEVICE)

        output_indices=[]
        while len(output_indices)<max_length:
            # dec_input: shape of [batch_size=1]
            _, hidden, output_index, stacks = dec(dec_input, hidden, stacks)
            if output_index.item()==EOS:
                break
            output_indices.append(output_index.item())
            dec_input = output_index.squeeze(0)

        return ' '.join([output_lang.index2word[output_index] for output_index in output_indices])

def eval_randomly(n=1):
    for i in range(n):
        pair = random.choice(pairs)
        with open(params.log_file, 'a+') as f:
            print('>', pair[0],file=f)
            print('=', pair[1],file=f)
            output_words = trans_one_sen(enc,dec,pair[0])
            print('<', output_words,file=f)
            print('',file=f)

def train_epochs():
    # enc_optim=optim.Adagrad(enc.parameters(),lr=LR)
    # dec_optim=optim.Adagrad(dec.parameters(),lr=LR)
    # enc_optim = optim.SGD(enc.parameters(), lr=LR)
    # dec_optim = optim.SGD(dec.parameters(), lr=LR)
    enc_optim = optim.Adam(enc.parameters(), lr=LR)
    dec_optim = optim.Adam(dec.parameters(), lr=LR)
    best_loss = None

    for epoch in range(NEPOCHS):
        epoch_start_time = time.time()
        loss = train(enc_optim, dec_optim, epoch)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            with open(params.enc_file, 'wb') as f:
                torch.save(enc, f)
            with open(params.dec_file, 'wb') as f:
                torch.save(dec, f)
        with open(params.log_file, 'a+') as f:
            print('end of epoch %d | time: %f s | loss: %f' %
                  (epoch,
                   time.time() - epoch_start_time,
                   loss),file=f)
            print('accuracy in training: ',
                  eval.train_accuracy(enc, dec),
                  file=f)


if __name__ == '__main__':
    # name='15254220464367697'
    # with open(params.enc_file, 'rb') as f:
    #     enc=torch.load(f,map_location='cpu')
    # with open(params.dec_file, 'rb') as f:
    #     dec=torch.load(f,map_location='cpu')

    train_epochs()
    # eval_randomly(100)


