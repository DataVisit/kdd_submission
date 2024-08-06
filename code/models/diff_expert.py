#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: Diff-expert
@Name: diff_expert.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from TrajUNet import UNet, WideAndDeep
from polyline_encoder import Polyline_Encoder
from map_encoder import Map_Encoder

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp1, inp2, inp3):
        # expand according to batch index so we can just split by _part_sizes
        inp_exp1 = inp1[self._batch_index]#.squeeze(1)
        inp_exp2 = inp2[self._batch_index]#.squeeze(1)
        inp_exp3 = inp3[self._batch_index].squeeze(1)
        return torch.split(inp_exp1, self._part_sizes, dim=0), torch.split(inp_exp2, self._part_sizes, dim=0), \
                torch.split(inp_exp3, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            self._nonzero_gates =self._nonzero_gates.unsqueeze(2).expand_as(stitched)
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),expert_out[-1].size(2), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
    

class Diff_Expert(nn.Module):
    def __init__(self, config):
        super(Diff_Expert, self).__init__()
        self.config = config
        self.ch = config.model.ch * 2
        self.attr_dim = config.model.attr_dim
        self.map_encoder = Map_Encoder(config)
        self.guide_emb = WideAndDeep(self.ch)
        self.poly_encoder = Polyline_Encoder(config)
        
        self.num_experts = 5
        self.k = 4
        self.noisy_gating = True

        self.input_size = self.ch*2
        self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.experts = nn.ModuleList([Model(config) for i in range(self.num_experts)])
        self.loss_coef=1e-2
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

   
    def forward(self, x, t, attr, interpolate_traj, image_tensor, ports, polylines):
        attr_emb = self.guide_emb(attr)
        image_emb = self.map_eocoder.generate(interpolate_traj, image_tensor)
        poly_emb = self.poly_encoder(ports, polylines).mean(dim=1).repeat(interpolate_traj.shape[0], 1)    
        guide_emb = torch.cat((attr_emb, image_emb, poly_emb), dim=1)
        
        gates, load = self.noisy_top_k_gating(guide_emb, self.training)
        choices = gates
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs_x, expert_inputs_t, expert_inputs_g = dispatcher.dispatch(x,t,guide_emb)
        gates = dispatcher.expert_to_gates()
        
        expert_outputs = [self.experts[i](expert_inputs_x[i], expert_inputs_t[i], expert_inputs_g[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss, choices 
