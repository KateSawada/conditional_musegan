# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from conditional_musegan.layers import GeneratorBlock, FinalGeneratorBlock

# A logger for this file
logger = getLogger(__name__)


class MuseganGenerator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(
        self,
        latent_dim,
        n_tracks,
        n_measures,
        measure_resolution,
        n_pitches,
        conditioning=False,
        conditioning_dim=0):
        """_summary_

        Args:
            latent_dim (int): Dimension of random noise
            n_tracks (int): Number of tracks
            n_measures (int): Number of Measures
            measure_resolution (int): time resolution per measure
            n_pitches (int): number of used pitches
            conditioning (bool, optional): Whether use conditioning. Defaults to False.
            conditioning_dim (int, optional): conditioning dimension. Defaults to 0.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        # self.conditioning = conditioning
        # self.conditioning_dim = conditioning_dim

        shared_conv_params = {
            "in_channel": [latent_dim, 256, 128, 64],
            "out_channel": [256, 128, 64, 32],
            "kernel": [(4, 1, 1), (1, 4, 1), (1, 1, 4), (1, 1, 3)],
            "stride": [(4, 1, 1), (1, 4, 1), (1, 1, 4), (1, 1, 1)],
        }
        self.shared_conv_network = nn.ModuleList()
        for i in range(len(shared_conv_params["in_channel"])):
            self.shared_conv_network += [
                GeneratorBlock(
                    shared_conv_params["in_channel"][i],
                    shared_conv_params["out_channel"][i],
                    shared_conv_params["kernel"][i],
                    shared_conv_params["stride"][i],
                )
            ]

        private_conv_params = {
            "in_channel": [32, 16],
            "out_channel": [16, 1],
            "kernel": [(1, 4, 1), (1, 1, 12)],
            "stride": [(1, 4, 1), (1, 1, 12)],
        }
        self.private_conv_network = nn.ModuleList()
        for i in range(len(private_conv_params["in_channel"]) - 1):
            self.private_conv_network += [
                nn.ModuleList(
                    [
                        GeneratorBlock(
                            private_conv_params["in_channel"][i],
                            private_conv_params["out_channel"][i],
                            private_conv_params["kernel"][i],
                            private_conv_params["stride"][i],
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]
        self.private_conv_network += [
            nn.ModuleList(
                    [
                        FinalGeneratorBlock(
                            private_conv_params["in_channel"][-1],
                            private_conv_params["out_channel"][-1],
                            private_conv_params["kernel"][-1],
                            private_conv_params["stride"][-1],
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
        ]

    def forward(self, x):
        # if (self.conditioning):
        #     x, condition = x
        #     condition = condition.view(-1, self.conditioning_dim)
        #     shape = list(x.shape)
        #     shape[1] = self.conditioning_dim
        #     condition = condition.expand(shape)
        #     x = torch.cat([x, condition], 1)
        # x = x.view(-1, self.latent_dim + self.conditioning_dim, 1, 1, 1)
        x = x.view(-1, self.latent_dim, 1, 1, 1)

        # shared
        for i in range(len(self.shared_conv_network)):
            x = self.shared_conv_network[i](x)

        # private (first)
        x = [conv(x) for conv in self.private_conv_network[0]]
        # private (second to final)
        for i in range(1, len(self.private_conv_network)):
            x = [conv(x_) for x_, conv in zip(x, self.private_conv_network[i])]
        x = torch.cat(x, 1)
        x = x.view(-1, self.n_tracks, self.n_measures * self.measure_resolution, self.n_pitches)
        x = torch.sigmoid(x)
        return x
