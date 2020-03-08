import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *


class MeanAggregator(nn.Module):
    def __init__(self, h_dim, dropout, seq_len=10):
        super(MeanAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

    def forward(self, s_hist, rel_hist, att_s_hist, self_att_s_hist, s, r,
                ent_embeds, ent_embeds_attribute, rel_embeds, W1, W3, W4):

        # for i in range(len(s_hist)):
        #     assert (len(s_hist[i]) == len(self_att_s_hist[i]))
        # print('forward')
        s_len_non_zero, s_tem, r_tem, embeds_stack, embeds_static_stack, len_s, s_idx = get_sorted_s_r_embed(
            s_hist, rel_hist, att_s_hist, s, r, ent_embeds,
            ent_embeds_attribute, rel_embeds, W1, W3, W4)
        # print('forward1')

        # To get mean vector at each time
        curr = 0
        rows = []
        cols = []
        for i, leng in enumerate(len_s):
            rows.extend([i] * leng)
            cols.extend(list(range(curr, curr + leng)))
            curr += leng
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        idxes = torch.stack([rows, cols], dim=0)

        mask_tensor = torch.sparse.FloatTensor(idxes, torch.ones(len(rows)))
        mask_tensor = mask_tensor.cuda()
        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)
        embeds_mean = embeds_sum / torch.Tensor(len_s).cuda().view(-1, 1)
        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())

        embeds_static_sum = torch.sparse.mm(mask_tensor, embeds_static_stack)
        embeds_static_mean = embeds_static_sum / torch.Tensor(
            len_s).cuda().view(-1, 1)
        embeds_static_split = torch.split(embeds_static_mean,
                                          s_len_non_zero.tolist())

        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len,
                                         3 * self.h_dim).cuda()
        att_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len,
                                           3 * self.h_dim).cuda()

        for i in range(len(embeds_split)):

            embeds = embeds_split[i]
            embeds_static = embeds_static_split[i]

            self_e_embed_static = ent_embeds[s_tem[i]].repeat(len(embeds), 1)
            self_e_embed_att = ent_embeds_attribute[s_tem[i]].repeat(
                len(embeds), 1)
            self_r_embed = rel_embeds[r_tem[i]].repeat(len(embeds), 1)

            self_att_embed = F.relu(
                W1(torch.tensor(self_att_s_hist[s_idx[i]]).cuda().view(-1, 1)))

            att_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                [self_att_embed, self_e_embed_att, embeds], dim=1)

            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                [self_e_embed_static, self_r_embed, embeds_static], dim=1)

        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
        att_embed_seq_tensor = self.dropout(att_embed_seq_tensor)

        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            s_embed_seq_tensor, s_len_non_zero, batch_first=True)
        att_packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            att_embed_seq_tensor, s_len_non_zero, batch_first=True)

        return s_packed_input, att_packed_input

    def predict(self, s_history, rel_history, att_history, self_att_history, s,
                r, ent_embeds, ent_embeds_attribute, rel_embeds, W1, W3, W4):
        inp_s = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        inp_att = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        for i, s_o in enumerate(s_history):
            r_o = rel_history[i]
            a_o = att_history[i]
            if type(a_o) != torch.Tensor:
                a_o = torch.tensor(a_o, dtype=torch.float)

            self_a_o = self_att_history[i]
            if type(self_a_o) != torch.Tensor:
                self_a_o = torch.tensor(self_a_o, dtype=torch.float)

            e_s_static = ent_embeds[s_o].view(-1, self.h_dim)
            e_s_att = ent_embeds_attribute[s_o].view(-1, self.h_dim)
            e_r = rel_embeds[r_o].view(-1, self.h_dim)
            e_att = F.relu(W1(a_o.type(torch.FloatTensor).cuda().view(
                -1, 1))).view(-1, self.h_dim)
            # e_s_att = F.relu(W2(torch.cat([e_att, e_s], dim=-1)))
            e = F.relu(W3(torch.cat([e_att, e_s_att, e_r], dim=-1)))
            e_static = F.relu(W4(torch.cat([e_s_static, e_r], dim=-1)))
            tem = torch.mean(e, dim=0)
            tem_static = torch.mean(e_static, dim=0)
            # print(self_a_o)
            e_self_att = F.relu(W1(self_a_o.cuda().view(1,
                                                        1))).view(self.h_dim)

            inp_s[i] = torch.cat([
                ent_embeds[s].view(self.h_dim), rel_embeds[r].view(self.h_dim),
                tem_static
            ],
                                 dim=0)
            inp_att[i] = torch.cat(
                [e_self_att, ent_embeds_attribute[s].view(self.h_dim), tem],
                dim=0)
        return inp_s, inp_att
