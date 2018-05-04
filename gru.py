from torch import nn
import torch
import numpy as np
import params
from params import args

SOS=params.SOS
EOS=params.EOS


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        # here input dimention is equal to hidden dimention
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,
                                      hidden_size,
                                      padding_idx=EOS)

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        # inputs: length * bsz
        # stacks: bsz * nstack * stacksz * stackelemsz
        embs = self.embedding(inputs)
        # inputs(length,bsz)->embd(length,bsz,embdsz)

        outputs,hidden=self.gru(embs,hidden)

        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size,
                                      hidden_size,
                                      padding_idx=EOS)

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, stacks):
        # input: shape of [bsz]
        # emb: 1 * bsz * embdsz
        emb = self.embedding(input).unsqueeze(0)


        # 1 * bsz * hsz -> bsz * hsz:
        output, hidden = self.gru(emb,hidden).squeeze(0)
        output = self.log_softmax(output,dim=1)
        # output: bsz * tar_vacabulary_size

        topv, topi = torch.topk(output,1,dim=1)
        output_index = topi
        # output_index: bsz * 1

        return output, hidden, output_index
