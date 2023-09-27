"""
Assumption:
    We have trained a model for the task of DTDG link prediction.
    Now, we want to used the trained model for the task of graph classification

July 6, 2023
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
  

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
  

class Runner(object):
    def __init__(self):
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(self.start_train, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))
        self.load_feature()
        logger.info('INFO: total length: {}, train length: {}, test length: {}'.format(self.len, len(self.train_shots), args.testlength))

        self.model = load_model(args).to(args.device)
        self.model_path = '../saved_models/{}/{}_{}_seed_{}.pth'.format(args.dataset, args.dataset,
                                                                   args.model, args.seed)
        logger.info("The model is going to be loaded from {}".format(self.model_path))
        self.model.load_state_dict(torch.load(self.model_path))

        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("INFO: Using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('INFO: using pre-defined feature')
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('INFO: using one-hot feature')
            args.nfeat = self.x.size(1)

    def tgclassification_test(self, epoch, readout_scheme):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_test_idx, t in enumerate(self.test_shots):
           self.model.eval()
           self.tgc_decoder.eval()
           with torch.no_grad():
              edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
              new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

              embeddings = self.model(edge_index, self.x)

              # graph readout
              tg_readout = readout_function(embeddings, readout_scheme)
              tg_embedding = torch.cat((tg_readout, 
                                        torch.from_numpy(self.t_graph_feat[t_test_idx + len(self.train_shots)]).to(args.device)))
          
              # graph classification
              tg_labels.append(self.t_graph_labels[t_test_idx + len(self.train_shots)].cpu().numpy())
              tg_preds.append(self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap
        

    def run(self):
        """
        Run the temporal graph classification task
        """
        readout_scheme = 'mean'
        graph_classifier_lr = 1e-4

        # load the TG-model
        self.model.eval()
        logger.info("Model is loaded.")
        self.model.init_hiddens()
        logger.info("Start training temporal graph classification.")

        # define optimizer and criterion
        optimizer = torch.optim.Adam(self.tgc_decoder.parameters(), lr=graph_classifier_lr)
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        decoder_loss_criterion = torch.nn.BCELoss()

        self.model.eval()
        self.tgc_decoder = self.tgc_decoder.train()
        
        t_total_start = time.time()
        min_loss = 10
        train_avg_epoch_loss_dict = {}
        for epoch in range(1, args.max_epoch + 1):
          t_epoch_start = time.time()
          epoch_losses = []
          for t_train_idx, t_train in enumerate(self.train_shots):
              optimizer.zero_grad()
              with torch.no_grad():
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t_train)
                embeddings = self.model(edge_index, self.x)
                self.model.update_hiddens_all_with(embeddings)
              
              # graph readout
              tg_readout = readout_function(embeddings, readout_scheme)
              tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t_train_idx]).to(args.device)))
            
              # graph classification
              tg_label = self.t_graph_labels[t_train_idx].float().view(1, )
              tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

              t_loss = decoder_loss_criterion(tg_pred, tg_label)
              t_loss.backward()
              optimizer.step()
              epoch_losses.append(t_loss.item())

          avg_epoch_loss = np.mean(epoch_losses)
          train_avg_epoch_loss_dict[epoch] = avg_epoch_loss

          patience = 0
          if avg_epoch_loss < min_loss:
                min_loss = avg_epoch_loss
                test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, readout_scheme)
                patience = 0
          else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:  # NOTE: args.min_epoch prevents it from stopping early in most cases
                    print('INFO: Early Stopping...')
                    break
          gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

          if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                          time.time() - t_epoch_start,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Test: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))
          
          if isnan(t_loss):
                print('ATTENTION: nan loss')
                break
          
        logger.info('>> Total time : %6.2f' % (time.time() - t_total_start))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # ------------ DEBUGGING ------------
        # save the training loss values
        partial_results_path = f'../data/output/log/{args.dataset}/{args.model}/'
        loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        with open(loss_log_filename, 'wb') as file:
            dump(train_avg_epoch_loss_dict, file)
        
        # plotting the training losses
        train_avg_epoch_loss_dict = load(open(loss_log_filename, 'rb'))
        train_values = train_avg_epoch_loss_dict.values()
        epoch_range = range(0, epoch)
        plt.plot(epoch_range, train_values, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(np.arange(0, epoch, 50))
        plt.legend(loc='best')
        plt.show()
        plt.savefig(f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.png')
        # -----------------------------------
        # -----------------------------------

        # Final Test
        test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, readout_scheme)
        logger.info("Final Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))


if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare

    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("INFO: Dataset: {}".format(args.dataset))
    data = loader(dataset=args.dataset, neg_sample=args.neg_sample)
    args.num_nodes = data['num_nodes']
    print("INFO: Number of nodes:", args.num_nodes)
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(args.seed) + '_log.txt')
    runner = Runner()
    runner.run()


# ----------------------
# commands to run:
# cd scripts
# python train_graph_classification.py --model=HTGN --seed=710  --dataset=aion