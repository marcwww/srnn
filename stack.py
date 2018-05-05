from torch import nn
import torch
import numpy as np
import params
from params import args

NACT = params.NACT
PUSH=params.PUSH
POP=params.POP
NOOP=params.NOOP
PAD=params.PAD
USE_STACK=args.use_stack
DEVICE=params.device
NONLINEAR=params.NONLINEAR

def create_stack(stack_size,stack_elem_size):
    return np.array([([EMPTY_VAL] * stack_elem_size)] * stack_size)

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

class EncoderSRNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 nstack, stack_depth, stack_size,
                 stack_elem_size):
        super(EncoderSRNN, self).__init__()
        # here input dimention is equal to hidden dimention
        self.hidden_size = hidden_size
        self.nstack=nstack
        self.stack_size=stack_size
        self.stack_depth=stack_depth
        self.stack_elem_size=stack_elem_size
        self.embedding = nn.Embedding(input_size,
                                      hidden_size,
                                      padding_idx=PAD)
        self.nonLinear=NONLINEAR()
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.hid2hid=nn.Linear(hidden_size,hidden_size)
        self.input2hid=nn.Linear(hidden_size,hidden_size)
        self.hid2act=[nn.Linear(hidden_size,NACT).to(DEVICE)
                      for _ in range(nstack)]
        self.hid2stack=[nn.Linear(hidden_size,stack_elem_size).to(DEVICE)
                        for _ in range(nstack)]
        self.stack2hid=[nn.Linear(stack_elem_size*stack_depth,hidden_size).to(DEVICE)
                        for _ in range(nstack)]

        self.softmax=nn.Softmax()
        self.empty_elem =torch.randn(1,self.stack_elem_size,requires_grad=True)

        W_up, W_down = shift_matrix(stack_size)
        self.W_up = torch.Tensor(W_up).to(DEVICE)

        self.W_down = torch.Tensor(W_down).to(DEVICE)

    def update_stack(self, stacks, si,
                     p_push, p_pop, p_noop, push_val):
        # stacks: bsz * nstack * stacksz * stackelemsz
        # new_elem: bsz * elemsz
        # push_val: bsz * elemsz
        batch_size=stacks.shape[0]
        stack=stacks[:,si,:,:].clone().to(DEVICE)

        p_push=p_push.unsqueeze(1)
        p_pop=p_pop.unsqueeze(1)
        p_noop=p_noop.unsqueeze(1)
        # stack: bsz * stacksz * stackelemsz
        stack= \
            p_push*(self.W_down.matmul(stack))+\
            p_pop*(self.W_up.matmul(stack))+\
            p_noop*stack

        p_push=p_push.squeeze(1)
        stack[:,0,:]=(p_push*push_val).clone()
        stack[:,self.stack_size-1,:]=\
            self.empty_elem.expand(batch_size,self.stack_elem_size).clone()

        return stack

    def forward(self, inputs, hidden, stacks):
        # inputs: length * bsz
        # stacks: bsz * nstack * stacksz * stackelemsz
        embs = self.embedding(inputs)
        # inputs(length,bsz)->embd(length,bsz,embdsz)

        batch_size=inputs.shape[1]

        outputs=[]
        for input in embs:
            # input: bsz * embdsz
            mid_hidden=self.input2hid(input)+self.hid2hid(hidden)

            if USE_STACK:
                # stack_vals: bsz * nstack * (stack_depth * stack_elem_size)
                # catenate all the readed vectors:
                stack_vals=stacks[:,:,:self.stack_depth,:].contiguous().\
                    view(batch_size,
                         self.nstack,
                         self.stack_depth*self.stack_elem_size).clone().to(DEVICE)

                # for each stack:
                for si in range(self.nstack):
                    # put each previous stack vals into the mid hidden:
                    mid_hidden+=self.stack2hid[si](stack_vals[:,si,:])

                    stacks=stacks.clone()
                    # using the current hidden to compute the actions:
                    # act: bsz * 3
                    act=self.hid2act[si](hidden)
                    p_push, p_pop, p_noop=act.chunk(NACT,dim=1)

                    # using the current hidden to compute the vals to push:
                    push_val=self.hid2stack[si](hidden)
                    push_val=self.nonLinear(push_val)

                    # update stack si:
                    stacks[:, si, :, :]=self.update_stack(stacks,si,
                                                       p_push,p_pop,p_noop,
                                                       push_val).clone()

            hidden=self.nonLinear(mid_hidden)
            output=hidden
            outputs.append(output)

        return outputs, hidden, stacks

    def init_stack(self,batch_size):
        return self.empty_elem.expand(batch_size,
                                               self.nstack,
                                               self.stack_size,
                                               self.stack_elem_size).contiguous()
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size,self.hidden_size)

class DecoderSRNN(nn.Module):
    def __init__(self, hidden_size, output_size,
                 nstack, stack_depth, stack_size,
                 stack_elem_size):
        super(DecoderSRNN, self).__init__()
        self.hidden_size = hidden_size

        self.nstack=nstack
        self.stack_size=stack_size
        self.stack_depth=stack_depth
        self.stack_elem_size=stack_elem_size
        self.embedding = nn.Embedding(output_size,
                                      hidden_size,
                                      padding_idx=PAD)
        self.nonLinear=NONLINEAR()
        # self.gru = nn.GRU(hidden_size, hidden_size)

        self.hid2hid = nn.Linear(hidden_size, hidden_size)
        self.input2hid = nn.Linear(hidden_size, hidden_size)
        self.hid2act = [nn.Linear(hidden_size, NACT).to(DEVICE)
                        for _ in range(nstack)]
        self.hid2stack = [nn.Linear(hidden_size, stack_elem_size).
                              to(DEVICE)
                            for _ in range(nstack)]
        self.stack2hid = [nn.Linear(stack_elem_size * stack_depth, hidden_size)
                              .to(DEVICE)
                          for _ in range(nstack)]
        self.hid2out = nn.Linear(hidden_size,output_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax=nn.Softmax()


        self.empty_elem = torch.randn(1, self.stack_elem_size,requires_grad=True)

        W_up, W_down = shift_matrix(stack_size)
        self.W_up = torch.Tensor(W_up).to(DEVICE)
        self.W_down = torch.Tensor(W_down).to(DEVICE)

        self.enc2dec=[nn.Linear(stack_elem_size,stack_elem_size).to(DEVICE)
                      for _ in range(nstack)]

    def update_stack(self, stacks, si,
                     p_push, p_pop, p_noop, push_val):
        # stacks: bsz * nstack * stacksz * stackelemsz
        # new_elem: bsz * elemsz
        # push_val: bsz * elemsz
        batch_size = stacks.shape[0]
        stack=stacks[:,si,:,:].clone().to(DEVICE)

        p_push=p_push.unsqueeze(1)
        p_pop=p_pop.unsqueeze(1)
        p_noop=p_noop.unsqueeze(1)
        # stack: bsz * stacksz * stackelemsz
        stack= \
            p_push*(self.W_down.matmul(stack))+\
            p_pop*(self.W_up.matmul(stack))+\
            p_noop*stack

        p_push=p_push.squeeze(1)
        stack[:,0,:]=(p_push*push_val).clone()
        stack[:,self.stack_size-1,:]=\
            (self.empty_elem.expand(batch_size,self.stack_elem_size)).clone()

        return stack

    def forward(self, input, hidden, stacks):
        # input: shape of [bsz]
        # stacks: bsz * nstack * stacksz * stackelemsz
        emb = self.embedding(input)
        # inputs(length,bsz)->embd(length,bsz,embdsz)

        batch_size = input.shape[0]

        # emb: bsz * embdsz

        mid_hidden = self.input2hid(emb) + self.hid2hid(hidden)

        if USE_STACK:
            # stack_vals: bsz * nstack * (stack_depth * stack_elem_size)
            # catenate all the readed vectors:
            stack_vals = stacks[:, :, :self.stack_depth, :].contiguous(). \
                view(batch_size,
                     self.nstack,
                     self.stack_depth * self.stack_elem_size).clone().to(DEVICE)

            # for each stack:
            for si in range(self.nstack):
                # put each previous stack vals into the mid hidden:
                mid_hidden += self.stack2hid[si](stack_vals[:, si, :])

                stacks = stacks.clone()
                # using the current hidden to compute the actions:
                # act: bsz * 3
                act = self.hid2act[si](hidden)
                p_push, p_pop, p_noop = act.chunk(NACT, dim=1)

                # using the current hidden to compute the vals to push:
                push_val = self.hid2stack[si](hidden)
                push_val = self.nonLinear(push_val)

                # update stack si:
                stacks[:, si, :, :] = self.update_stack(stacks, si,
                                                        p_push, p_pop, p_noop,
                                                        push_val).clone()

        hidden = self.nonLinear(mid_hidden)
        output = self.hid2out(hidden)
        output = self.log_softmax(output)
        # output: bsz * tar_vacabulary_size

        top1 = output.data.max(1)[1]
        top1 = top1.unsqueeze(1)
        output_index=top1
        # output_index: bsz * 1

        return output, hidden, output_index, stacks
