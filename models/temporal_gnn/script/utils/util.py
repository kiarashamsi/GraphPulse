import random
import os
import torch
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx



logger = logging.getLogger()
disease_path = '../data/input/raw/disease/disease_lp.feats.npz'


def mkdirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def write_to_text(args, results, seed):
    if args.to_compute_results:
        with open(args.result_txt, "a") as f:
            data = str(seed) + '\t' + '\t'.join(['{:.6f}'.format(result) for result in results[1:]]) + '\n'
            f.write(data)
    else:
        print('Not write to file')


def compute_mean_std(args):
    if args.to_compute_results:
        results = np.loadtxt(args.result_txt, delimiter='\t')
        results = results[:, 1:]
        mean_list = np.mean(results, axis=0)
        std_list = np.std(results, axis=0)
        with open(args.result_txt, "a") as f:
            data = 'results: ' + '\t'.join(
                ['{:.2f}+{:.2f}'.format(mean * 100, std * 100) for mean, std in zip(mean_list, std_list)]) + '\n'
            f.write(data)
    else:
        print('Not write to file')


def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('INFO: fixed random seed')


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def filters(edge_index, threshold):
    return np.where(np.where(edge_index >= threshold, 1, 0).sum(axis=0) == 0)


def neg_sampling_basic(edge_index, num_nodes):
    import random
    num_neg_samples = edge_index.shape[1]
    idx = edge_index[0] * num_nodes + edge_index[1]
    rng = range(num_nodes ** 2)
    perm = torch.tensor(random.sample(rng, num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]
    row, col = torch.true_divide(perm, num_nodes), perm % num_nodes
    return torch.stack([row, col], dim=0).long().numpy()


def negative_sampling(adjs, cum_nodes):
    '''
    negative sampling each time stamp without sampling previous time
    '''

    history_perm = []
    neg_adjs = []
    for t in range(len(adjs)):
        edge_index = adjs[t]
        num_nodes = len(cum_nodes[t])
        num_neg_samples = edge_index.shape[1]  # get number of negative sampling nodes
        current_perm = edge_index[0] * num_nodes + edge_index[1]  # obtain the total hash slots
        idx = np.concatenate([current_perm, history_perm])  # expand the slots with historical slots
        rng = range(num_nodes ** 2)  # get range of negative sampling
        perm = torch.tensor(random.sample(rng, num_neg_samples))  # get negative perm
        mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
        rest = mask.nonzero().view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.tensor(random.sample(rng, rest.size(0)))
            mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
            perm[rest] = tmp
            rest = rest[mask.nonzero().view(-1)]
        # row, col = perm / num_nodes, perm % num_nodes
        row, col = torch.true_divide(perm, num_nodes), perm % num_nodes
        neg_adjs.append(torch.stack([row, col], dim=0).long())
        history_perm += list(perm.numpy())
    return neg_adjs


def total_pairwise_similarity(x):
    # x = F.normalize(x)
    z = torch.matmul(x, x.transpose(0, 1))
    n = z.size(1) * (z.size(1) - 1)  # A_2^n
    total_similarity = torch.sum(z) - torch.sum(z.diagonal())
    average = total_similarity / n
    return 1 - average


class MLP(torch.nn.Module):
  """
  Binary Classifier to work as a decoder

  reference:
    https://github.com/twitter-research/tgn/blob/master/utils/utils.py
  """
  def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, drop=0.5):
    super().__init__()
    self.fc_1 = torch.nn.Linear(in_dim, hidden_dim_1)
    self.fc_2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
    self.fc_3 = torch.nn.Linear(hidden_dim_1, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


def readout_function(embeddings, readout_scheme='mean'):
    """
    Read out function to generate a representation for the whole graph
    reference:    
    https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification/blob/master/readouts/basic_readout.py
    """
    # note: x.size(): [#nodes, args.n_out]
    if readout_scheme == 'max':
      readout = torch.max(embeddings, dim=0)[0].squeeze() # max readout
    elif readout_scheme == 'mean':
      readout = torch.mean(embeddings, dim=0).squeeze() # mean readout
    elif readout_scheme == 'sum':
      readout = torch.sum(embeddings, dim=0).squeeze() # sum readout
    else:
      readout = None
      raise ValueError("Readout Method Undefined!")
    return readout
  

def extra_dataset_attributes_loading(args):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    partial_path = f'../data/input/raw/{args.dataset}/'
   
    # load graph lables
    label_filename = f'{partial_path}/{args.dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)

    # load and process graph-pooled (node-level) features 
    edgelist_filename = f'{partial_path}/{args.dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    TG_feats = []
    for ts in uniq_ts_list:
       ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
       ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight', create_using=nx.MultiDiGraph)
       node_list = list(ts_G.nodes)
       indegree_list = np.array(ts_G.in_degree(node_list))
       weighted_indegree_list = np.array(ts_G.in_degree(node_list, weight='weight'))
       outdegree_list = np.array(ts_G.out_degree(node_list))
       weighted_outdegree_list = np.array(ts_G.out_degree(node_list, weight='weight'))

       TG_feats.append(np.array([np.sum(indegree_list), np.sum(weighted_indegree_list), 
                                np.sum(outdegree_list), np.sum(weighted_outdegree_list)]))
    
    # scale the temporal graph features to have a reasonable range
    scalar = MinMaxScaler()
    TG_feats = scalar.fit_transform(TG_feats)

    return TG_labels, TG_feats