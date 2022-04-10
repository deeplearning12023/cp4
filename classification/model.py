import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def logits(self, source_tokens, segment_ids, **unused):
        # ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def get_loss(self, source_tokens, segment_ids, targets, **unused):
        logits = self.logits(source_tokens, segment_ids)
        loss = F.cross_entropy(logits, targets)
        return loss
