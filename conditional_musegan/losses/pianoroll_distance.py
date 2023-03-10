# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class PianorollDistanceLoss(nn.Module):
    def __init__(
        self,
        loss_type="L2"
        ):
        super(PianorollDistanceLoss, self).__init__()

        assert loss_type  in ["l2", "l1", "bce"], f"{loss_type} is not supported."

        if loss_type == "l2":
            self.distance_function = F.mse_loss
        elif loss_type == "l1":
            self.distance_function = F.l1_loss
        elif loss_type == "bce":
            self.distance_function = F.binary_cross_entropy

    def fowrard(self, x1, x2):
        return self.distance_function(x1, x2)
