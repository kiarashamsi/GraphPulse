import os
import sys
import random
import time
import math
import torch
import numpy as np
from math import isnan
import torch.optim as optim
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from script.inits import prepare
from script.models.load_model import load_model
from script.loss import ReconLoss, VGAEloss
from script.config import args
from script.utils.util import init_logger, logger
from script.utils.data_util import loader,prepare_dir
from script.utils.util import MLP, readout_function, extra_dataset_attributes_loading
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import matplotlib.pyplot as plt


log_interval = 50


class Runner(object):
    def __init__(self):
        self.readout_scheme = 'mean'
        self.tgc_lr = 1e-4

        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))

        # load features
        if args.trainable_feat:
            self.x = None
            logger.info("using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                feature = sp.load_npz('../data/input/raw/disease/disease_lp.feats.npz').toarray()
                self.x = [torch.from_numpy(feature).float().to(args.device)] * len(self.train_shots)
                logger.info('using pre-defined feature')
            else:
                self.x = [torch.eye(args.num_nodes).to(args.device)] * len(self.train_shots)
                logger.info('using one-hot feature')
            args.nfeat = self.x[0].size(1)

        # load the temporal graph embedding model
        self.model = load_model(args).to(args.device)

        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 

        logger.info('tota length: {}, train length: {}, test length: {}'.format(self.len, len(self.train_shots), args.testlength))


    def train(self):
        print('1. Initialization')
        minloss = 10
        min_epoch = 1
        max_patience = args.patience
        patience = 0
        # define optimizer and criterion
        optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )
        criterion = torch.nn.BCELoss()

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()
    
        train_avg_epoch_loss_dict = {}

        print("2. Start training")
        t_total_start = time.time()
        # best_results = [0] * 5
        for epoch in range(1, args.max_epoch + 1):
            self.model.train()
            self.tgc_decoder.train()
            t_epoch_start = time.time()
            epoch_losses = []
            self.model.init_hiddens()
            optimizer.zero_grad()
            embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in self.train_shots], self.x)
            self.model.update_hiddens_all_with(embeddings[-1])
            # compute loss
            for t, z in enumerate(embeddings):
                # edge_index = prepare(data, t)[0]
                # epoch_loss = self.loss(z, edge_index)
                pos_edge_index, neg_edge_index = prepare(data, t)[1], prepare(data, t)[2]
                # epoch_loss = self.loss(z, pos_edge_index, neg_edge_index)
                # epoch_losses.append(epoch_loss)

                # graph readout
                tg_readout = readout_function(z, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t]).to(args.device)))
                
                # graph classification
                tg_label = self.t_graph_labels[t].float().view(1, )
                tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()
                epoch_loss = criterion(tg_pred, tg_label)
                epoch_losses.append(epoch_loss)

            sum(epoch_losses).backward()
            optimizer.step()

            # update the best results.
            average_epoch_loss = np.mean([epoch_loss.item() for epoch_loss in epoch_losses])
            train_avg_epoch_loss_dict[epoch] = average_epoch_loss

            if average_epoch_loss < minloss:
                minloss = average_epoch_loss
                test_epoch, test_auc, test_ap = self.tgclassification_test(epoch)
                patience = 0
            else:
                patience += 1
                if epoch > min_epoch and patience > max_patience:
                    print('early stopping')
                    break
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            if epoch % args.log_interval == 0:
                logger.info('==' * 27)
                logger.info("Epoch:{}, Loss: {:.3f}, Time: {:.2f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss,
                                                                                          time.time() - t_epoch_start,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Epoch:{:}, Best Test: AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))

            if isnan(epoch_loss):
                break
        logger.info('>> Total time : %6.2f' % (time.time() - t_total_start))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (
            args.lr, args.nhid, args.nb_window))
        
        # Final Test
        test_epoch, test_auc, test_ap = self.tgclassification_test(epoch)
        logger.info("Final Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))    

    def tgclassification_test(self, epoch):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in self.test_shots], self.x)

        for t_test, z in zip(self.test_shots, embeddings):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():

                # graph readout
                tg_readout = readout_function(z, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t_test]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_test].cpu().numpy())
                tg_preds.append(self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        if epoch % args.log_interval == 0:
            logger.info(
                '\tEpoch:{}: Test AUC: {:.4f}; Test AP: {:.4f}'.format(epoch, auc, ap))
        return epoch, auc, ap


if __name__ == '__main__':
    random.seed(args.seed)  # random seed
    data = loader(dataset=args.dataset)  # enron10, fb, dblp
    args.num_nodes = data['num_nodes']
    log_folder = prepare_dir(args.output_folder)  # 2.create folder
    init_logger(log_folder + args.dataset + '_seed_' + str(args.seed) + '.txt')
    run = Runner()
    run.train()
