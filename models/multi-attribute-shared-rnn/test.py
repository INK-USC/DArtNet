import argparse
import numpy as np
import torch
import utils
import os
from model import DArtNet
import pickle
import collections
import time

result = collections.namedtuple(
    "result",
    ["epoch", "MRR", "sub_att_loss", "MR", "Hits1", "Hits3", "Hits10"])

result_dict = {}


def test(args):
    # load data
    num_nodes, num_rels, num_att = utils.get_total_number(
        args.dataset_path, 'stat.txt')
    test_data, test_times = utils.load_hexaruples(args.dataset_path,
                                                  'test.txt')
    total_data, total_times = utils.load_hexaruples(args.dataset_path,
                                                    'train.txt', 'valid.txt',
                                                    'test.txt')

    model_dir = 'models/' + args.dataset + '/{}-{}-{}-{}'.format(
        args.dropout, args.n_hidden, args.gamma, args.num_k)
    model_state_file = model_dir + '/epoch-{}.pth'.format(args.epoch)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(999)

    model = DArtNet(num_nodes,
                    args.n_hidden,
                    num_rels,
                    num_att,
                    model=args.model,
                    seq_len=args.seq_len,
                    num_k=args.num_k,
                    gamma=args.gamma).float()

    if use_cuda:
        model.cuda()

    test_sub_entity = '/test_entity_s_history_data.txt'
    test_sub_rel = '/test_rel_s_history_data.txt'
    test_sub_att = '/test_att_s_history_data.txt'
    test_sub_self_att = '/test_self_att_s_history_data.txt'

    test_ob_entity = '/test_entity_o_history_data.txt'
    test_ob_rel = '/test_rel_o_history_data.txt'
    test_ob_att = '/test_att_o_history_data.txt'
    test_ob_self_att = '/test_self_att_o_history_data.txt'

    with open(args.dataset_path + test_sub_entity, 'rb') as f:
        entity_s_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_sub_rel, 'rb') as f:
        rel_s_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_sub_att, 'rb') as f:
        att_s_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_sub_self_att, 'rb') as f:
        self_att_s_history_data_test = pickle.load(f)

    with open(args.dataset_path + test_ob_entity, 'rb') as f:
        entity_o_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_ob_rel, 'rb') as f:
        rel_o_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_ob_att, 'rb') as f:
        att_o_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_ob_self_att, 'rb') as f:
        self_att_o_history_data_test = pickle.load(f)

    print(f'\nstart testing model file : {model_state_file}')

    checkpoint = torch.load(model_state_file,
                            map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    model.init_history()

    model.latest_time = checkpoint['latest_time']

    print("Using epoch: {}".format(checkpoint['epoch']))

    total_data = torch.from_numpy(total_data)
    test_data = torch.from_numpy(test_data)

    model.eval()
    total_att_sub_loss = 0
    total_ranks = np.array([])
    total_ranks_filter = np.array([])
    ranks = []

    with torch.no_grad():
        latest_time = test_times[0]
        j = 0
        while j < len(test_data):
            k = j
            while k < len(test_data):
                if test_data[k][-1] == test_data[j][-1]:
                    k += 1
                else:
                    break

            start = j
            while start < k:
                end = min(k, start + args.batch_size)

                batch_data = test_data[start:end].clone()
                s_hist = entity_s_history_data_test[start:end].copy()
                o_hist = entity_o_history_data_test[start:end].copy()
                rel_s_hist = rel_s_history_data_test[start:end].copy()
                rel_o_hist = rel_o_history_data_test[start:end].copy()
                att_s_hist = att_s_history_data_test[start:end].copy()
                att_o_hist = att_o_history_data_test[start:end].copy()
                self_att_s_hist = self_att_s_history_data_test[start:end].copy(
                )
                self_att_o_hist = self_att_o_history_data_test[start:end].copy(
                )

                if use_cuda:
                    batch_data = batch_data.cuda()

                loss_sub = model.predict(batch_data, s_hist, rel_s_hist,
                                         att_s_hist, self_att_s_hist, o_hist,
                                         rel_o_hist, att_o_hist,
                                         self_att_o_hist)

                total_att_sub_loss += (loss_sub.item() * (end - start + 1))

                start += args.batch_size

            for i in range(j, k):
                batch_data = test_data[i].clone()
                s_hist = entity_s_history_data_test[i].copy()
                o_hist = entity_o_history_data_test[i].copy()
                rel_s_hist = rel_s_history_data_test[i].copy()
                rel_o_hist = rel_o_history_data_test[i].copy()
                att_s_hist = att_s_history_data_test[i].copy()
                att_o_hist = att_o_history_data_test[i].copy()
                self_att_s_hist = self_att_s_history_data_test[i].copy()
                self_att_o_hist = self_att_o_history_data_test[i].copy()

                if use_cuda:
                    batch_data = batch_data.cuda()

                ranks_pred = model.evaluate_filter(batch_data, s_hist,
                                                   rel_s_hist, att_s_hist,
                                                   self_att_s_hist, o_hist,
                                                   rel_o_hist, att_o_hist,
                                                   self_att_o_hist, total_data)

                total_ranks_filter = np.concatenate(
                    (total_ranks_filter, ranks_pred))

            j = k

    ranks.append(total_ranks_filter)

    for rank in ranks:
        total_ranks = np.concatenate((total_ranks, rank))
    mrr = np.mean(1.0 / total_ranks)
    mr = np.mean(total_ranks)
    hits = []

    for hit in [1, 3, 10]:
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)

        print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
    print("MRR (filtered): {:.6f}".format(mrr))
    print("MR (filtered): {:.6f}".format(mr))
    print("test att sub Loss: {:.6f}".format(total_att_sub_loss /
                                             (len(test_data))))

    result_epoch = result(epoch=args.epoch,
                          MRR=100 * mrr,
                          sub_att_loss=total_att_sub_loss / len(test_data),
                          MR=mr,
                          Hits1=100 * hits[0],
                          Hits3=100 * hits[1],
                          Hits10=100 * hits[2])

    result_dict[args.epoch] = result_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DArtNet')
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="dropout probability")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        help="dataset_path to use")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--n-hidden",
                        type=int,
                        default=200,
                        help="number of hidden units")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k",
                        type=int,
                        default=10,
                        help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--epoch", type=int, required=True)

    args = parser.parse_args()

    model_dir = 'models/' + args.dataset + '/{}-{}-{}-{}'.format(
        args.dropout, args.n_hidden, args.gamma, args.num_k)

    try:
        with open(model_dir + '/compiled_results_test.tsv', 'r') as f:
            res = f.readlines()[1:]

        for r in res:
            a = r.strip().split('\t')
            result_dict[int(a[0])] = result(epoch=int(a[0]),
                                            MRR=float(a[1]),
                                            sub_att_loss=float(a[5]),
                                            MR=float(a[6]),
                                            Hits1=float(a[2]),
                                            Hits3=float(a[3]),
                                            Hits10=float(a[4]))
    except FileNotFoundError as _:
        pass

    try:

        epoch = args.epoch
        print(f'testing epoch {epoch}')
        model_state_file = model_dir + '/epoch-{}.pth'.format(args.epoch)
        flag = True
        test(args)

        with open(model_dir + '/compiled_results_test.tsv', 'w') as f:
            f.write('Epoch\tMRR\tHits1\tHits3\tHits10\tAttribute_Loss\tMR\n')
            for key, val in result_dict.items():

                f.write(
                    f'{key}\t{val.MRR}\t{val.Hits1}\t{val.Hits3}\t{val.Hits10}\t{val.sub_att_loss}\t{val.MR}\n'
                )

    except KeyboardInterrupt as _:
        with open(model_dir + '/compiled_results_test.tsv', 'w') as f:
            f.write('Epoch\tMRR\tHits1\tHits3\tHits10\tAttribute_Loss\tMR\n')
            for key, val in result_dict.items():

                f.write(
                    f'{key}\t{val.MRR}\t{val.Hits1}\t{val.Hits3}\t{val.Hits10}\t{val.sub_att_loss}\t{val.MR}\n'
                )
