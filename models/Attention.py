# -*- encoding: utf-8 -*-
import numpy as np
import random
import torch.nn as nn
import torch
import copy


class Attention_1(nn.Module):
    """
        Attention-based fusion.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: tagert embbeding, shape: (batch_size, embed_dim)
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, with_ave=1, mul=False, common_type="one_layer"):
        """
        :param embed_dim:
        :param with_ave: if 'attention_noAve' == True: 之后加上平均值
        :param mul: 是否 global 与 local 相乘。
        common_type: "one_layer"
        """
        super().__init__()
        self.with_ave = with_ave
        self.mul = mul
        self.embed_dim = embed_dim
        if "two_layer" in common_type:
            self.embedding_common = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1)
            )
        elif "one_layer" in common_type:
            self.embedding_common = nn.Sequential(
                nn.Linear(embed_dim, 1)
            )

        self.common_type = common_type

        self.softmax = nn.Softmax(dim=1)
        self.weights = 0  # attention 权重
        self.global_emb_weight_net = nn.Linear(1, 1, False)  # 存储 raw_global_emb 的权重
        self.change_raw_global_emb_weight(1)

    def get_raw_global_emb_weight(self):
        """
        得到 global_emb 的权重
        :return:
        """
        return self.global_emb_weight_net.weight.item()

    def change_raw_global_emb_weight(self, new_value: float):
        self.global_emb_weight_net.weight.data.fill_(new_value)

    def get_attention_weight(self):
        return torch.tensor(self.weights).clone().detach().cpu()

    def forward(self, local_embs: torch.Tensor):
        # local_emb: batch, embedding_size, num_emb
        local_embs = local_embs.permute(0, 2, 1)  # batch, num_emb, embedding_size
        weights = self.embedding_common(local_embs).squeeze(2)
        weights = self.softmax(weights)
        self.weights = weights
        if self.with_ave:
            raw_global_weight = self.get_raw_global_emb_weight()
            self.weights = 0.5 * weights + raw_global_weight * 0.5 / weights.shape[1]  # weights + meanpooling
        return self.weights
