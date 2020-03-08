import numpy as np
import os
import torch
import torch.nn.functional as F


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_hexaruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            att_head = float(line_split[3])
            att_tail = float(line_split[4])
            time = int(line_split[5])
            quadrupleList.append([head, rel, tail, att_head, att_tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                att_head = float(line_split[3])
                att_tail = float(line_split[4])
                time = int(line_split[5])
                quadrupleList.append(
                    [head, rel, tail, att_head, att_tail, time])
                times.add(time)
    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                att_head = float(line_split[3])
                att_tail = float(line_split[4])
                time = int(line_split[5])
                quadrupleList.append(
                    [head, rel, tail, att_head, att_tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[5])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[5])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[5])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def make_batch(a, b, c, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n], b[i:i + n], c[i:i + n]


def make_batch2(a, b, c, d, e, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n], b[i:i + n], c[i:i + n], d[i:i + n], e[i:i + n]


def make_batch3(a, b, c, d, e, f, g, h, w, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n], b[i:i +
                            n], c[i:i +
                                  n], d[i:i +
                                        n], e[i:i +
                                              n], f[i:i +
                                                    n], g[i:i +
                                                          n], h[i:i +
                                                                n], w[i:i + n]


def get_data(s_hist, o_hist):
    data = None
    for i, s_his in enumerate(s_hist):
        if len(s_his) != 0:
            # print(s_his)
            tem = torch.cat((torch.LongTensor([i]).repeat(
                len(s_his), 1), torch.LongTensor(s_his.cpu())),
                            dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)

    for i, o_his in enumerate(o_hist):
        if len(o_his) != 0:
            tem = torch.cat((torch.LongTensor(o_his[:, 1].cpu()).view(
                -1, 1), torch.LongTensor(o_his[:, 0].cpu()).view(
                    -1, 1), torch.LongTensor([i]).repeat(len(o_his), 1)),
                            dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)
    data = np.unique(data, axis=0)
    return data


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


'''
Get sorted s and r to make batch for RNN (sorted by length)
'''


def get_sorted_s_r_embed(s_hist, rel_hist, att_hist, s, r, ent_embeds,
                         ent_embeds_attribute, rel_embeds, W1, W3, W4):
    s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    rel_hist_sorted = []
    att_hist_sorted = []
    for idx in s_idx:
        s_hist_sorted.append(s_hist[idx.item()])
        rel_hist_sorted.append(rel_hist[idx.item()])
        att_hist_sorted.append(att_hist[idx.item()])

    flat_s = []
    flat_rel = []
    flat_att = []
    len_s = []

    s_hist_sorted = s_hist_sorted[:num_non_zero]
    rel_hist_sorted = rel_hist_sorted[:num_non_zero]
    att_hist_sorted = att_hist_sorted[:num_non_zero]

    for i in range(len(s_hist_sorted)):
        for j in range(len(s_hist_sorted[i])):
            len_s.append(len(s_hist_sorted[i][j]))
            for k in range(len(s_hist_sorted[i][j])):
                flat_s.append(s_hist_sorted[i][j][k])
                flat_rel.append(rel_hist_sorted[i][j][k])
                flat_att.append(att_hist_sorted[i][j][k])

    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds_s_static = ent_embeds[torch.LongTensor(flat_s).cuda()]
    embeds_s_att = ent_embeds_attribute[torch.LongTensor(flat_s).cuda()]
    embeds_rel = rel_embeds[torch.LongTensor(flat_rel).cuda()]
    embeds_att = F.relu(W1(torch.tensor(flat_att).view(-1, 1).cuda()))

    embeds = F.relu(
        W3(torch.cat([embeds_att, embeds_s_att, embeds_rel], dim=1)))
    # embeds_split = torch.split(embeds, len_s)

    embeds_static = F.relu(W4(torch.cat([embeds_s_static, embeds_rel], dim=1)))
    # embeds_static_split = torch.split(embeds_static, len_s)

    return s_len_non_zero, s_tem, r_tem, embeds, embeds_static, len_s, s_idx  # embeds_split, embeds_static_split,
