################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length

        # Parameters of g-gate (modulation)
        self.W_gx = nn.Parameter(torch.empty(input_dim, num_hidden, device=device))
        self.W_gh = nn.Parameter(torch.empty(num_hidden, num_hidden, device=device))
        self.b_g = nn.Parameter(torch.empty(num_hidden, device=device))

        # Parameters of i-gate (input)
        self.W_ix = nn.Parameter(torch.empty(input_dim, num_hidden, device=device))
        self.W_ih = nn.Parameter(torch.empty(num_hidden, num_hidden, device=device))
        self.b_i = nn.Parameter(torch.empty(num_hidden, device=device))

        # Parameters of f-gate (forget)
        self.W_fx = nn.Parameter(torch.empty(input_dim, num_hidden, device=device))
        self.W_fh = nn.Parameter(torch.empty(num_hidden, num_hidden, device=device))
        self.b_f = nn.Parameter(torch.empty(num_hidden, device=device))

        # Parameters of o-gate (output)
        self.W_ox = nn.Parameter(torch.empty(input_dim, num_hidden, device=device))
        self.W_oh = nn.Parameter(torch.empty(num_hidden, num_hidden, device=device))
        self.b_o = nn.Parameter(torch.empty(num_hidden, device=device))

        self.W_ph = nn.Parameter(torch.empty(num_hidden, num_classes, device=device))
        self.b_p = nn.Parameter(torch.empty(num_classes, device=device))

        for W in [self.W_gx, self.W_gh,\
                  self.W_ix, self.W_ih,\
                  self.W_fx, self.W_fh,\
                  self.W_ox, self.W_oh,\
                  self.W_ph]: nn.init.xavier_uniform_(W)

        nn.init.ones_(self.b_f)
        for b in [self.b_g, self.b_i, self.b_o,\
                  self.b_p]: nn.init.zeros_(b)

        self.h_init = torch.zeros(self.W_gh.size(0), requires_grad=True)

    def forward(self, x):
        # Implementation here ...

        #h = torch.zeros(self.W_gh.size(0), requires_grad=True)
        h = self.h_init
        c = torch.zeros(self.W_gh.size(0))

        if self.W_gx.size(0) == self.b_p.size(0):
            x = F.one_hot(x.long()).float()
        else:
            x = x[:,:,None]

        for t in range(self.seq_length):
            g = torch.tanh(x[:,t]@self.W_gx + h@self.W_gh + self.b_g)
            i = torch.sigmoid(x[:,t]@self.W_ix + h@self.W_ih + self.b_i)
            f = torch.sigmoid(x[:,t]@self.W_fx + h@self.W_fh + self.b_f)
            o = torch.sigmoid(x[:,t]@self.W_ox + h@self.W_oh + self.b_o)
            c = g*i + c*f
            h = torch.tanh(c*o)

        out = h@self.W_ph + self.b_p

        return out
