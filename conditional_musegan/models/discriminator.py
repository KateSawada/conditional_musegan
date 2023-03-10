# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from conditional_musegan.layers import DiscriminatorBlock


# A logger for this file
logger = getLogger(__name__)


class MuseganDiscriminator(nn.Module):
    """A convolutional neural network (CNN) based discriminator. The
    discriminator takes as input either a real sample (in the training data) or
    a fake sample (generated by the generator) and outputs a scalar indicating
    its authentity.
    """
    def __init__(
        self,
        n_tracks,
        n_measures,
        measure_resolution,
        n_pitches,
        conditioning=False,
        conditioning_dim=0):
        """_summary_

        Args:
            n_tracks (int): Number of tracks
            n_measures (int): Number of Measures
            measure_resolution (int): time resolution per measure
            n_pitches (int): number of used pitches
            conditioning (bool, optional): Whether use conditioning. Defaults to False.
            conditioning_dim (int, optional): conditioning dimension. Defaults to 0.
        """
        super().__init__()

        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        # self.conditioning = conditioning
        # self.conditioning_dim = conditioning_dim

        private_conv_params = {
            "in_channel": [1, 16],
            "out_channel": [16, 16],
            "kernel": [(1, 1, 12), (1, 4, 1)],
            "stride": [(1, 1, 12), (1, 4, 1)],
        }

        self.private_conv_network = nn.ModuleList()
        for i in range(len(private_conv_params["in_channel"])):
            self.private_conv_network += [
                nn.ModuleList(
                    [
                        DiscriminatorBlock(
                            private_conv_params["in_channel"][i],
                            private_conv_params["out_channel"][i],
                            private_conv_params["kernel"][i],
                            private_conv_params["stride"][i],
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        self.shared_conv_network = nn.ModuleList()
        shared_conv_params = {
            "in_channel": [16 * n_tracks, 64, 64, 128, 128],
            "out_channel": [64, 64, 128, 128, 256],
            "kernel": [(1, 1, 3), (1, 1, 4), (1, 4, 1), (2, 1, 1), (3, 1, 1)],
            "stride": [(1, 1, 1), (1, 1, 4), (1, 4, 1), (1, 1, 1), (3, 1, 1)]
        }

        for i in range(len(shared_conv_params["in_channel"])):
            self.shared_conv_network += [
                DiscriminatorBlock(
                    shared_conv_params["in_channel"][i],
                    shared_conv_params["out_channel"][i],
                    shared_conv_params["kernel"][i],
                    shared_conv_params["stride"][i],
                )
            ]

        self.dense = torch.nn.Linear(256, 1)


    def forward(self, x):
        x = x.view(-1, self.n_tracks, self.n_measures, self.measure_resolution, self.n_pitches)
        x = [x[:, i].view(-1, 1, self.n_measures, self.measure_resolution, self.n_pitches) for i in range(self.n_tracks)]

        # private
        for i in range(len(self.private_conv_network)):
            # x = [conv(x_) for x_, conv in zip(x, self.private_conv_network[i])]
            x = [conv(x[j]) for j, conv in enumerate(self.private_conv_network[i])]
        x = torch.cat(x, 1)

        # shared
        for i in range(len(self.shared_conv_network)):
            x = self.shared_conv_network[i](x)

        x = x.view(-1, 256)
        x = self.dense(x)
        return x