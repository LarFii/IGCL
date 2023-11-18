import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as torch_sp

from time import time
from reckit import timer
from reckit import randint_choice
from model.base import AbstractRecommender
from util.common import ensureDir
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.pytorch import sp_mat_to_sp_tensor
from data import PairwiseSamplerV2

class _IGCL(nn.Module):
    def __init__(
            self, 
            num_users, 
            num_items, 
            embed_dim, 
            n_layers, 
            recon_dim, 
            rnoise_flag, 
            rnoise_eps
        ):
        super(_IGCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.rnoise_flag = rnoise_flag
        self.rnoise_eps = rnoise_eps

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(embed_dim, recon_dim)
        self.fc2 = nn.Linear(recon_dim, embed_dim)
    
    def _perturb_embedding(self, embeds):
        noise = (F.normalize(torch.rand_like(embeds).cuda(), dim=-1) * torch.sign(embeds)) * self.rnoise_eps
                
        return embeds + noise

    def gcn(self, norm_adj, user_embeddings, item_embeddings):
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            
            if self.rnoise_flag:
                ego_embeddings = self._perturb_embedding(ego_embeddings)

            all_embeddings += [ego_embeddings]
    
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings
    
    def encode(self, x):         
        return self.fc1(x)

    def decode(self, z):
        return self.fc2(z)
    
    def loss_func(self, recon_x, x):
        MSE_loss = nn.MSELoss(reduction='sum')
        recon_loss = MSE_loss(recon_x, x)

        return recon_loss

    def forward(self, norm_adj, user_embeddings, item_embeddings):
        user_embeddings, item_embeddings = self.gcn(norm_adj, user_embeddings, item_embeddings)
        z_user, z_item = self.encode(user_embeddings), self.encode(item_embeddings)
        user_embeddings_gen, item_embeddings_gen = self.decode(z_user), self.decode(z_item)

        loss_user = self.loss_func(user_embeddings_gen, user_embeddings)
        loss_item = self.loss_func(item_embeddings_gen, item_embeddings)
        loss = loss_user + loss_item

        return user_embeddings_gen, item_embeddings_gen, loss


class _LightGCN(nn.Module):
    def __init__(
            self, 
            num_users, 
            num_items, 
            embed_dim, 
            norm_adj, 
            n_layers, 
            IGCL_layers, 
            recon_dim, 
            rnoise_flag,
            rnoise_eps
        ):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.IGCL_layers = IGCL_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        self.IGCL = _IGCL(
                        num_users, 
                        num_items, 
                        embed_dim, 
                        IGCL_layers, 
                        recon_dim, 
                        rnoise_flag, 
                        rnoise_eps
                    )    

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).to('cuda')
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).to('cuda')
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, pd_graph, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)   
   
        user_embeddings_gen, item_embeddings_gen, recon_loss = \
            self.IGCL(pd_graph, user_embeddings, item_embeddings)

        user_embeddings1 = F.normalize(user_embeddings, dim=1)
        item_embeddings1 = F.normalize(item_embeddings, dim=1)

        user_embeddings_gen = F.normalize(user_embeddings_gen, dim=1)
        item_embeddings_gen = F.normalize(item_embeddings_gen, dim=1)  


        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)

        user_embs2 = F.embedding(users, user_embeddings_gen)
        item_embs2 = F.embedding(items, item_embeddings_gen)   

        sup_pos_ratings = inner_product(user_embs, item_embs)       
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   
        sup_logits = sup_pos_ratings - sup_neg_ratings              

        pos_ratings_user = inner_product(user_embs1, user_embs2)    
        pos_ratings_item = inner_product(item_embs1, item_embs2)    

        tot_ratings_user = torch.matmul(user_embs1, 
                                        torch.transpose(user_embeddings_gen, 0, 1))        
        tot_ratings_item = torch.matmul(item_embs1, 
                                        torch.transpose(item_embeddings_gen, 0, 1))
        
        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  

        return sup_logits, ssl_logits_user, ssl_logits_item, recon_loss

    def _gcn(self, ego_embeddings, layers, norm_adj):
        all_embeddings = [ego_embeddings]

        for k in range(layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
    
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings
    
    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        
        return self._gcn(ego_embeddings, self.n_layers, norm_adj)
    
    def sparse_dropout(self, mat, dropout):
        if dropout == 0.0:
            return mat
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()

        return torch.sparse.FloatTensor(indices, values, size)

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)

class IGCL(AbstractRecommender):
    def __init__(self, config):
        super(IGCL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # gen_2eral hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.rnoise_flag = config["rnoise_flag"]
        self.rnoise_eps = config["rnoise_eps"]

        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_temp = config["ssl_temp"]

        self.recon_reg = config["recon_reg"]
        self.IGCL_layers = config['IGCL_layers']
        self.recon_dim = config['recon_dim']

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '-ratio=%.1f-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_temp,
            self.ssl_reg
        )

        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s.pth' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, 
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(
            self.num_users, 
            self.num_items, 
            self.emb_size, 
            adj_matrix, 
            self.n_layers, 
            self.IGCL_layers, 
            self.recon_dim,
            self.rnoise_flag,
            self.rnoise_eps
        ).to(self.device)

        self.lightgcn.reset_parameters(init_method=self.param_init)
            
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
            user_np = np.array(users_np)[keep_idx]
            item_np = np.array(items_np)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)    
   
        stopping_step = 0

        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss, total_recon_loss, total_infonce_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            training_start_time = time()
            pd_graph = []
            for _ in range(0, self.n_layers):
                tmp_graph = self.create_adj_mat(is_subgraph=True)
                pd_graph.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))


            self.lightgcn.train()

            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)

                
                sup_logits, ssl_logits_user, ssl_logits_item, recon_loss = self.lightgcn(
                    pd_graph, bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)

                loss = bpr_loss + self.reg * reg_loss + self.ssl_reg * infonce_loss + self.recon_reg * recon_loss 

                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                total_infonce_loss += self.ssl_reg * infonce_loss
                total_recon_loss += self.recon_reg * recon_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
            
            self.logger.info("[iter %d : loss : %f = %f + %f + %f + %f, time: %f]" % (
                epoch, 
                total_loss / self.num_ratings,
                total_bpr_loss / self.num_ratings,
                total_infonce_loss / self.num_ratings,
                total_reg_loss / self.num_ratings,
                total_recon_loss / self.num_ratings,
                time()-training_start_time,))
            
            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")

                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        
                elif epoch > 50:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break
        
        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)
    
    # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[0] < current_result[0]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()