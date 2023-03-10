# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(GradientPenaltyLoss, self).__init__()

    def forward(self, discriminator, real_samples, fake_samples):
        """Compute the gradient penalty for regularization. Intuitively, the
        gradient penalty help stabilize the magnitude of the gradients that the
        discriminator provides to the generator, and thus help stabilize the
        training of the generator.
        """
        # Get random interpolations between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
        interpolates = interpolates.requires_grad_(True)
        # Get the discriminator output for the interpolations
        d_interpolates = discriminator(interpolates)
        # Get gradients w.r.t. the interpolations
        fake = torch.ones(real_samples.size(0), 1).to(real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
