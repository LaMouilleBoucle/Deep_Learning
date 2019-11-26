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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length

        self.W_hx = nn.Parameter(torch.empty(input_dim, num_hidden, device=device))
        self.W_hh = nn.Parameter(torch.empty(num_hidden, num_hidden, device=device))
        self.W_ph = nn.Parameter(torch.empty(num_hidden, num_classes, device=device))
        self.b_h = nn.Parameter(torch.empty(num_hidden, device=device))
        self.b_p = nn.Parameter(torch.empty(num_classes, device=device))

        for W in [self.W_hx, self.W_hh, self.W_ph]: nn.init.xavier_uniform_(W)
        for b in [self.b_h, self.b_p]: nn.init.zeros_(b)

        self.h_init = torch.zeros(self.W_hh.size(0), requires_grad=True)

    def forward(self, x):
        # Implementation here ...

        h = self.h_init
        #h = torch.zeros(self.W_hh.size(0), requires_grad=True)
        if self.W_hx.size(0) == self.b_p.size(0):
            x = F.one_hot(x.long()).float()
        else:
            x = x[:,:,None]

        for t in range(self.seq_length):
            h = torch.tanh(x[:,t]@self.W_hx + h@self.W_hh + self.b_h)

        # if self.W_hx.size(0) == self.b_p.size(0):
        #     for t in range(self.seq_length):
        #         x_one_hot = F.one_hot(x[:,t].long(), self.b_p.size(0)).float()
        #         h = torch.tanh(x_one_hot@self.W_hx + h@self.W_hh + self.b_h)
        # else:
        #     for t in range(self.seq_length):
        #         x_dense = x[:,t].view(-1, 1)
        #         h = torch.tanh(x_dense@self.W_hx + h@self.W_hh + self.b_h)

        out = h@self.W_ph + self.b_p

        return out
