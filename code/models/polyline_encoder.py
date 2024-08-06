#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Diff-expert
@Name: preprocess.py
"""
from math import sin, cos, sqrt, atan2, radians, asin
import numpy as np
import torch


def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T


def time_warping(x, length=200):
    """
    Resamples a trajectory to a new length.
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    warped_trajectory = np.zeros((2, length))
    for i in range(2):
        warped_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return warped_trajectory.T


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather consts for $t$ and reshape to feature map shape
    :param consts: (N, 1, 1)
    :param t: (N, H, W)
    :return: (N, H, W)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0, t, alpha_bar):
    # get mean and var of xt given x0
    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1 - gather(alpha_bar, t)
    # sample xt from q(xt | x0)
    eps = torch.randn_like(x0).to(x0.device)
    xt = mean + (var ** 0.5) * eps
    return xt, eps  # also returns noise


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta, t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next
