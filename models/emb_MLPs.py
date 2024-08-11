import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Attention import Attention_1
from loss import pearson_loss, huber_loss, kl_loss
import Dense_layer


def min_max_norm(a, eps=1e-12, dim=-1, min_value=0, max_value=1):
    # Min-Max scaling
    min_a = torch.min(a, dim=dim, keepdim=True).values
    max_a = torch.max(a, dim=dim, keepdim=True).values
    n = (a - min_a) / (max_a - min_a + eps) *(max_value-min_value) + min_value
    return n


def get_dense_layer(args, dense_type, input_dim, field_num, mlp_dims, dropout, cross_num=3):
    embed_dim = int(input_dim / field_num)
    if dense_type == 'MultiLayerPerceptron':
        dense_layer =  Dense_layer.MultiLayerPerceptronNet(input_dim=input_dim,
                             embed_dims=mlp_dims, output_layer=True, dropout=dropout)
    elif dense_type == 'FM':
        dense_layer = Dense_layer.FM(field_num=field_num, embed_dim=embed_dim)
    elif dense_type == 'DeepFM':
        dense_layer = Dense_layer.DeepFM(mlp_dims=mlp_dims, field_num=field_num, embed_dim=embed_dim, dropout=dropout)
    elif dense_type == 'DeepCrossNet':
        dense_layer = Dense_layer.DeepCrossNet(cross_num=cross_num, mlp_dims=mlp_dims, field_num=field_num, embed_dim=embed_dim, dropout=dropout)
    elif dense_type == 'InnerProductNet':
        dense_layer = Dense_layer.InnerProductNet(mlp_dims=mlp_dims, field_num=field_num, embed_dim=embed_dim, dropout=dropout)

    return dense_layer

def get_controller(args, controller_type, field_dims, embed_dim):
    if controller_type == 'controller_mlp':
        controller = controller_mlp(args, input_dim=len(field_dims) * embed_dim,
                                         embed_dims=[len(field_dims)])
    elif controller_type == 'MvFS':
        controller = MvFS_Controller(input_dim=len(field_dims) * embed_dim,
                                     embed_dim=len(field_dims))
    elif controller_type == 'attention':
        controller = Attention_1(embed_dim, with_ave=args.AttentionWithAve, mul=False,
                                      common_type='one_layer')

    return controller

class EMB(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x, kmax_index=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``

        :return: (batch, embedding_size, num_emb)
        """
        if kmax_index is not None:
            return self.forward_select(x, kmax_index)

        x = (x + x.new_tensor(self.offsets).unsqueeze(0)).long()
        return self.embedding(x).transpose(1,2)

    def forward_select(self, x, kmax_index):
        select_x = (x + x.new_tensor(self.offsets).unsqueeze(0)).long()

        # ********************selection *************************
        batch_size, seq_len = select_x.shape
        kmax_len = kmax_index.shape[1]
        # 生成批次索引
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, kmax_len)
        # 使用高级索引进行筛选
        select_x = select_x[batch_indices, kmax_index]

        return self.embedding(select_x).transpose(1, 2)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x


class MvFS_Controller(nn.Module):
    """
    Controller in MvFS
    """
    class SelectionNetwork(nn.Module):
        def __init__(self, input_dims, output_dims):
            super().__init__()

            self.mlp = MultiLayerPerceptron(input_dim=input_dims,
                                            embed_dims=[output_dims], output_layer=False, dropout=0.0)
            self.weight_init(self.mlp)

        def forward(self, input_mlp):
            output_layer = self.mlp(input_mlp)
            return torch.softmax(output_layer, dim=1)

        def weight_init(self, m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __init__(self, input_dim, embed_dim, num_selections=4):
        """

        :param input_dim:
        :param embed_dim:
        :param num_selections: num of selection network
        """
        super().__init__()
        self.inputdim = input_dim
        self.num_selections = num_selections

        self.T = 1

        self.gate = nn.Sequential(nn.Linear(embed_dim * num_selections, num_selections))

        self.SelectionNetworks = nn.ModuleList(
            [MvFS_Controller.SelectionNetwork(input_dim, embed_dim) for i in range(num_selections)]
        )

    def forward(self, emb_fields):

        input_mlp = emb_fields.flatten(start_dim=1).float()
        importance_list = []
        for i in range(self.num_selections):
            importance_vector = self.SelectionNetworks[i](input_mlp)
            importance_list.append(importance_vector)

        gate_input = torch.cat(importance_list, 1)
        selection_influence = self.gate(gate_input)
        selection_influence = torch.sigmoid(selection_influence)

        scores = None
        for i in range(self.num_selections):
            score = torch.mul(importance_list[i], selection_influence[:, i].unsqueeze(1))
            if i == 0:
                scores = score
            else:
                scores = torch.add(scores, score)

        scores = 0.5 * (1 + torch.tanh(self.T * (scores - 0.1)))

        if self.T < 5:
            self.T += 0.001
        return scores


class controller_mlp(nn.Module):
    def __init__(self, args, input_dim, embed_dims):
        super().__init__()
        self.inputdim = input_dim
        self.mlp = MultiLayerPerceptron(input_dim=self.inputdim,
                                        embed_dims=embed_dims, output_layer=False, dropout=args.dropout)
        self.weight_init(self.mlp)
    
    def forward(self, emb_fields):
        input_mlp = emb_fields.flatten(start_dim=1).float()
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class AdaFS_soft(nn.Module): 
    def __init__(self,args):
        super().__init__()
        self.num = len(args.field_dims)
        self.embed_dim = args.embed_dim
        self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = MultiLayerPerceptron(input_dim=len(args.field_dims)*self.embed_dim,
                                        embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)
        self.controller = controller_mlp(args, input_dim=len(args.field_dims)*self.embed_dim, embed_dims=[len(args.field_dims)])
        self.weight = 0
        self.useBN = args.useBN
        self.UseController = args.controller
        self.BN = nn.BatchNorm1d(self.embed_dim)
        self.stage = -1


        self.softmaxAddNormType = args.softmaxAddNorm


    def forward(self, field):
        field = self.emb(field)
        #对每个feature进行batchnorm
        if self.useBN == True:
            field = self.BN(field)
        if self.UseController and self.stage == 1:
            self.weight = self.controller(field)
            if self.softmaxAddNormType.split("_")[0] == 'Max-min':
                min_value, max_value = float(self.softmaxAddNormType.split("_")[1]), float(self.softmaxAddNormType.split("_")[2])
                self.weight = min_max_norm(self.weight, min_value=min_value, max_value=max_value)
            field = field * torch.unsqueeze(self.weight,1)        
        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        return torch.sigmoid(res.squeeze(1))

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0] 
    return index, x.gather(dim, index)

class AdaFS_hard(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.criterion = torch.nn.BCELoss()
        self.darts_frequency = args.darts_frequency

        self.num = len(args.field_dims)
        self.embed_dim = args.embed_dim
        self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = MultiLayerPerceptron(input_dim=len(args.field_dims)*self.embed_dim,
                                        embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)
        if args.model_name == 'AdaFS_hard':
            self.controller = controller_mlp(args, input_dim=len(args.field_dims) * self.embed_dim,
                                             embed_dims=[len(args.field_dims)])
        elif args.model_name == 'AdaFS_hard_attention':
            self.controller = Attention_1(self.embed_dim, with_ave=args.AttentionWithAve, mul=False, common_type='one_layer')

            self.controller1 = controller_mlp(args, input_dim=len(args.field_dims) * self.embed_dim,
                                             embed_dims=[len(args.field_dims)])

        self.UseController = args.controller
        self.BN = nn.BatchNorm1d(self.embed_dim)
        self.k = args.k
        self.useWeight = args.useWeight 
        self.reWeight = args.reWeight
        self.useBN = args.useBN
        self.device = args.device
        self.stage = -1

        self.threshold = float(args.AdaFS_hard_threshold)  # 默认为 -1

        self.optimizer, self.optimizer_model, self.optimizer_darts = self.init_optimizer(args)

    def init_optimizer(self, args):
        model_name = args.model_name
        learning_rate = args.learning_rate
        learning_rate_darts = args.learning_rate_darts
        weight_decay = args.weight_decay
        model = self

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if model_name == 'NoSlct':
            optimizer_model = None
            optimizer_darts = None
        else:
            optimizer_model = torch.optim.Adam(
                params=[param for name, param in model.named_parameters() if 'controller' not in name],
                lr=learning_rate,
                weight_decay=weight_decay)
            optimizer_darts = torch.optim.Adam(
                params=[param for name, param in model.named_parameters() if 'controller' in name],
                lr=learning_rate_darts,
                weight_decay=weight_decay)
        return optimizer, optimizer_model, optimizer_darts


    def threshold_mask(self, weight):
        threshold = 1.0/(weight.shape[-1]*self.threshold)
        choose_mask = (weight >= threshold).float()
        # **防止出现一个 weight 都不选择的情况**
        k=1
        topk_values, topk_indices = torch.topk(weight, k=k, dim=1)
        # 使用索引将对应位置的元素赋值为1
        choose_mask = choose_mask.scatter_(1, topk_indices, 1)  # dim, induce, set_value
        weight_new = weight*choose_mask
        # **防止出现一个 weight 都不选择的情况 end**
        if self.reWeight == True:
            weight_new = weight_new / torch.sum(weight_new, dim=1).unsqueeze(1)  # reweight, 使结果和为1
        if self.useWeight:
            pass  # 填充对应索引位置为weight值
        else:
            weight_new = (weight_new>0.0).float() # 对应索引位置填充1
        return weight_new

    def forward(self, field, target, step):
        """
        field = self.emb(field)  # (batch, embedding_size, num_emb)

        # 得到权重 (batch, num_emb)
        weight = self.controller(field)
        # 得到 topk_weight 和 index
        topK_index, topK_weight = kmax_pooling(weight, self.k)

        # 创建跟 weight 同维度的 mask，topK_index 位赋予 weight 值，其余为0
        mask = torch.zeros(weight.shape[0],weight.shape[1])
        mask = mask.scatter_(1, topK_index, topK_weight)

        # 和 field 进行相乘，对应 field
        field = field * torch.unsqueeze(mask,1)

        """
        field = self.emb(field)  # 得到16维度 embedding
        # 对每个feature进行batchnorm
        if self.useBN == True:
            field = self.BN(field)
        if self.UseController and self.stage == 1:
            weight = self.controller(field)
            if self.threshold > 0:
                mask = self.threshold_mask(weight)
            else:
                topK_index, topK_weight = kmax_pooling(weight, 1, self.k)
                if self.reWeight == True:
                    topK_weight = topK_weight/torch.sum(topK_weight,dim=1).unsqueeze(1) # reweight, 使结果和为1
                # 创建跟weight同维度的mask，index位赋予值，其余为0
                mask = torch.zeros(weight.shape[0],weight.shape[1]).to(self.device)
                if self.useWeight:
                    mask = mask.scatter_(1,topK_index,topK_weight) # 填充对应索引位置为weight值
                else:
                    mask = mask.scatter_(1,topK_index,torch.ones(topK_weight.shape[0],topK_weight.shape[1])) #对应索引位置填充1

            field = field * torch.unsqueeze(mask,1)      
        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        res = torch.sigmoid(res.squeeze(1))

        loss = self.criterion(res, target)


        self.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        # Update all params of model if do not use controller
        if not self.UseController:
            self.optimizer.step()
        # pretrain
        if self.UseController and self.stage == 0:
            self.optimizer_model.step()
        # search stage, alternatively update main RS network and Darts weights
        if self.UseController and self.stage == 1:
            self.optimizer_model.step()
            if (step + 1) % self.darts_frequency == 0:
                self.optimizer_darts.step()

        return loss

    def predict(self, field):
        field = self.emb(field)  # 得到16维度 embedding
        # 对每个feature进行batchnorm
        if self.useBN == True:
            field = self.BN(field)
        if self.UseController and self.stage == 1:
            weight = self.controller(field)
            if self.threshold > 0:
                mask = self.threshold_mask(weight)
            else:
                kmax_index, kmax_weight = kmax_pooling(weight, 1, self.k)
                if self.reWeight == True:
                    kmax_weight = kmax_weight / torch.sum(kmax_weight, dim=1).unsqueeze(1)  # reweight, 使结果和为1
                # 创建跟weight同维度的mask，index位赋予值，其余为0
                mask = torch.zeros(weight.shape[0], weight.shape[1]).to(self.device)
                if self.useWeight:
                    mask = mask.scatter_(1, kmax_index, kmax_weight)  # 填充对应索引位置为weight值
                else:
                    mask = mask.scatter_(1, kmax_index,
                                         torch.ones(kmax_weight.shape[0], kmax_weight.shape[1]))  # 对应索引位置填充1

            field = field * torch.unsqueeze(mask, 1)
        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        res = torch.sigmoid(res.squeeze(1))
        return res

class AEFS_emb_align_addLoss(nn.Module):
    """
    20231226
    """
    def __init__(self, args, sub_class=False):
        super().__init__()
        if sub_class:
            return
        self.criterion = torch.nn.BCELoss()
        self.darts_frequency = args.darts_frequency

        self.num = len(args.field_dims)
        self.UseController = args.controller
        assert self.UseController

        # normal model
        self.embed_dim = args.embed_dim
        self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = MultiLayerPerceptron(input_dim=len(args.field_dims)*self.embed_dim,
                                        embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)
        if 'AdaFS_twoModel' in args.model_name:
            self.controller = controller_mlp(args, input_dim=len(args.field_dims) * self.embed_dim,
                                             embed_dims=[len(args.field_dims)])
        elif 'twoModel_attention' in args.model_name:
            self.controller = Attention_1(self.embed_dim, with_ave=args.AttentionWithAve, mul=False, common_type='one_layer')
        self.BN = nn.BatchNorm1d(self.embed_dim)

        # small model
        self.embed_dim_small = args.embed_dim_small
        self.emb_small = EMB(args.field_dims[:self.num],self.embed_dim_small)
        if 'AdaFS_twoModel' in args.model_name:
            self.controller_small = controller_mlp(args, input_dim=len(args.field_dims) * self.embed_dim_small,
                                             embed_dims=[len(args.field_dims)])
        elif 'twoModel_attention' in args.model_name:
            self.controller_small = Attention_1(self.embed_dim_small, with_ave=args.AttentionWithAve, mul=False, common_type='one_layer')
        self.BN_small = nn.BatchNorm1d(self.embed_dim_small)
        self.mlp_smallToNormalEmb = nn.Linear(self.embed_dim_small, self.embed_dim)

        self.k = args.k
        self.useWeight = args.useWeight
        self.reWeight = args.reWeight
        self.useBN = args.useBN
        self.device = args.device
        self.stage = -1

        self.threshold = float(args.AdaFS_hard_threshold)  # 默认为 -1

        self.optimizer_all, self.optimizer_normal, \
            self.optimizer_controller, self.optimizer_small = self.init_optimizer(args)

        # two_model_optimizer_type
        self.two_model_optimizer_type = args.two_model_optimizer_type
        assert self.two_model_optimizer_type == 'simultaneous'
        self.two_model_controller_position = args.two_model_controller_position

        self.field_align_loss = self.init_align_loss(args.field_align_loss)
        self.score_align_loss = self.init_align_loss(args.score_align_loss)
        temp = args.score_and_field_align_loss_weight.split("_")
        self.score_align_loss_w, self.field_align_loss_w = float(temp[0]), float(temp[1])

    def init_align_loss(self, loss_type):
        if loss_type == 'pearson':
            return pearson_loss
        elif loss_type == 'huber':
            return huber_loss
        elif loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif loss_type == 'MSE':
            temp = torch.nn.MSELoss(reduction='mean')
            return temp
        elif loss_type == 'kl_loss':
            return kl_loss
    def init_optimizer(self, args):
        model_name = args.model_name
        learning_rate = args.learning_rate
        learning_rate_darts = args.learning_rate_darts
        weight_decay = args.weight_decay
        model = self

        optimizer_all = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if model_name == 'NoSlct':
            optimizer_model = None
            optimizer_controller = None
        else:
            params_optimizer_model = []
            params_optimizer_darts = []
            params_optimizer_small = []

            for name, param in model.named_parameters():
                if 'controller' in name:
                    params_optimizer_darts.append(param)
                elif 'small' in name:
                    params_optimizer_small.append(param)
                else:
                    params_optimizer_model.append(param)

            optimizer_model = torch.optim.Adam(
                params=params_optimizer_model, lr=learning_rate, weight_decay=weight_decay)
            optimizer_controller = torch.optim.Adam(
                params=params_optimizer_darts, lr=learning_rate_darts, weight_decay=weight_decay)
            optimizer_small = torch.optim.Adam(
                params=params_optimizer_small, lr=learning_rate, weight_decay=weight_decay)
        return optimizer_all, optimizer_model, optimizer_controller, optimizer_small

    def threshold_mask(self, weight):
        threshold = 1.0/(weight.shape[-1]*self.threshold)
        choose_mask = (weight >= threshold).float()
        # **防止出现一个 weight 都不选择的情况**
        k=1
        topk_values, topk_indices = torch.topk(weight, k=k, dim=1)
        # 使用索引将对应位置的元素赋值为1
        choose_mask = choose_mask.scatter_(1, topk_indices, 1)  # dim, induce, set_value
        weight_new = weight*choose_mask
        # **防止出现一个 weight 都不选择的情况 end**
        if self.reWeight == True:
            weight_new = weight_new / torch.sum(weight_new, dim=1).unsqueeze(1)  # reweight, 使结果和为1
        if self.useWeight:
            pass  # 填充对应索引位置为weight值
        else:
            weight_new =  (weight_new>0.0).float() # 对应索引位置填充1
        return weight_new

    def get_predict_score(self, field, emb_layer=None, controller=None, mlp_layer=None, BN_layer=None,
                          emb_layer_samllToNormal=None, controller_position='small_emb'):
        if (emb_layer is None) and (controller is None) and (mlp_layer is None):
            emb_layer = self.emb
            controller = self.controller
            mlp_layer = self.mlp
            BN_layer = self.BN
        normal_field_from_small = None
        field = emb_layer(field)  # 得到16维度 embedding
        # 对每个feature进行batchnorm
        if self.useBN == True:
            field = BN_layer(field)
        if controller_position == 'normal_emb':
            field = emb_layer_samllToNormal(field.transpose(1, 2)).transpose(1, 2)
            normal_field_from_small = field.clone()

        weight = controller(field)
        if self.threshold > 0:
            mask = self.threshold_mask(weight)
        else:
            kmax_index, kmax_weight = kmax_pooling(weight,1,self.k)
            if self.reWeight == True:
                kmax_weight = kmax_weight/torch.sum(kmax_weight,dim=1).unsqueeze(1) # reweight, 使结果和为1
            # 创建跟weight同维度的mask，index位赋予值，其余为0
            mask = torch.zeros(weight.shape[0],weight.shape[1]).to(self.device)
            if self.useWeight:
                mask = mask.scatter_(1, kmax_index, kmax_weight) # 填充对应索引位置为weight值
            else:
                mask = mask.scatter_(1, kmax_index, torch.ones(kmax_weight.shape[0],kmax_weight.shape[1])) #对应索引位置填充1

        if controller_position == 'small_emb':
            field = emb_layer_samllToNormal(field.transpose(1, 2)).transpose(1, 2)
            normal_field_from_small = field.clone()

        field = field * torch.unsqueeze(mask,1)


        input_mlp = field.flatten(start_dim=1).float()
        res = mlp_layer(input_mlp)
        res = torch.sigmoid(res.squeeze(1))
        return res, mask, normal_field_from_small
    def get_predict_score_normal(self, field, emb_layer, mlp_layer, BN_layer, emb_layer_samllToNormal=None):
        """
        不进行 feature selection
        :param field:
        :param emb_layer:
        :param mlp_layer:
        :return:
        """
        field = emb_layer(field)  # 得到相同维度 embedding
        # 对每个feature进行batchnorm
        if self.useBN == True:
            field = BN_layer(field)
        if emb_layer_samllToNormal is not None:
            field = emb_layer_samllToNormal(field.transpose(1,2)).transpose(1,2)
        input_mlp = field.flatten(start_dim=1).float()
        res = mlp_layer(input_mlp)
        res = torch.sigmoid(res.squeeze(1))
        return res

    def get_predict_score_with_mask(self, field, mask):
        field = self.emb(field)  # 得到16维度 embedding
        # 对每个feature进行batchnorm
        if self.useBN == True:
            field = self.BN(field)

        field = field * torch.unsqueeze(mask,1)
        returned_normal_field = field.clone()

        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        res = torch.sigmoid(res.squeeze(1))
        return res, returned_normal_field

    def pretrain(self, field, target):
        # pretrain
        res1 = self.get_predict_score_normal(
            field, emb_layer=self.emb, mlp_layer=self.mlp, BN_layer=self.BN)
        loss_normal = self.criterion(res1, target)
        res2 = self.get_predict_score_normal(
            field, emb_layer=self.emb_small, mlp_layer=self.mlp, BN_layer=self.BN_small,
            emb_layer_samllToNormal=self.mlp_smallToNormalEmb)
        loss_small = self.criterion(res2, target)

        (loss_small+loss_normal).backward()
        self.optimizer_normal.step()
        self.optimizer_small.step()

        return loss_small+loss_normal

    def forward(self, field, target, step):
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection mode for 'NAN' 'inf'

        self.zero_grad()

        if self.stage == 0:
            return self.pretrain(field, target)

        controller_position = self.two_model_controller_position
        if controller_position == 'small_emb':
            controller = self.controller_small
        elif controller_position == 'normal_emb':
            controller = self.controller
        result_small, mask, normal_field_from_small = self.get_predict_score(
            field, emb_layer=self.emb_small, controller=controller, mlp_layer=self.mlp,
        BN_layer=self.BN_small, emb_layer_samllToNormal=self.mlp_smallToNormalEmb, controller_position=controller_position)
        loss_small = self.criterion(result_small, target)

        result, normal_field = self.get_predict_score_with_mask(field, mask)
        loss_normal = self.criterion(result, target)

        score_align_loss = self.score_align_loss(result_small, result.detach())
        field_align_loss = self.field_align_loss((normal_field_from_small * torch.unsqueeze(mask,1)).transpose(1,2),
                                                 normal_field.detach().transpose(1,2))
        loss = loss_normal+loss_small + \
               self.score_align_loss_w * score_align_loss + self.field_align_loss_w * field_align_loss
        # search stage, alternatively update main RS network and Darts weights
        loss.backward()
        self.optimizer_normal.step()
        self.optimizer_small.step()
        if self.stage == 1 and ((step + 1) % self.darts_frequency == 0):
            self.optimizer_controller.step()

        return loss

    def predict(self, field):
        if self.stage == 0:
            # pretrain
            result = self.get_predict_score_normal(
                field, emb_layer=self.emb, mlp_layer=self.mlp, BN_layer=self.BN)

        else:
            controller_position = self.two_model_controller_position
            if controller_position == 'small_emb':
                controller = self.controller_small
            elif controller_position == 'normal_emb':
                controller = self.controller
            result_small, mask, normal_field_from_small = self.get_predict_score(
                field, emb_layer=self.emb_small, controller=controller, mlp_layer=self.mlp,
            BN_layer=self.BN_small, emb_layer_samllToNormal=self.mlp_smallToNormalEmb, controller_position=controller_position)
            result, returned_normal_field = self.get_predict_score_with_mask(field, mask)
        return result

class AEFS(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.criterion = torch.nn.BCELoss()
        self.darts_frequency = args.darts_frequency

        self.num = len(args.field_dims)
        self.field_dims = args.field_dims
        self.UseController = args.controller
        assert self.UseController

        # normal model
        self.embed_dim = args.embed_dim
        self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = get_dense_layer(args, args.dense_type, input_dim=len(args.field_dims)*self.embed_dim,
                                   field_num=len(args.field_dims), mlp_dims=args.mlp_dims, dropout=args.dropout)

        self.controller = get_controller(args, args.controller_type, args.field_dims, args.embed_dim)


        self.BN = nn.BatchNorm1d(self.embed_dim)

        # small model
        self.embed_dim_small = args.embed_dim_small
        self.emb_small = EMB(args.field_dims[:self.num],self.embed_dim_small)

        self.controller_small = get_controller(args, args.controller_type, args.field_dims, args.embed_dim_small)
        self.BN_small = nn.BatchNorm1d(self.embed_dim_small)
        self.mlp_smallToNormalEmb = nn.Linear(self.embed_dim_small, self.embed_dim)
        self.mlp_small = get_dense_layer(args, args.dense_type, input_dim=len(args.field_dims)*self.embed_dim_small,
                                   field_num=len(args.field_dims), mlp_dims=args.mlp_dims, dropout=args.dropout)
        self.small_loss_weight = float(args.small_loss_weight)

        self.k = args.k
        self.useWeight = args.useWeight
        self.reWeight = args.reWeight
        self.useBN = args.useBN
        self.device = args.device
        self.stage = -1

        self.threshold = float(args.AdaFS_hard_threshold)  # 默认为 -1

        self.optimizer_all, self.optimizer_normal, \
            self.optimizer_controller, self.optimizer_small = self.init_optimizer(args)

        # two_model_optimizer_type
        self.two_model_optimizer_type = args.two_model_optimizer_type
        assert self.two_model_optimizer_type == 'simultaneous'
        self.two_model_controller_position = args.two_model_controller_position

        self.field_align_loss = self.init_align_loss(args.field_align_loss)
        self.score_align_loss = self.init_align_loss(args.score_align_loss)
        temp = args.score_and_field_align_loss_weight.split("_")
        self.score_align_loss_w, self.field_align_loss_w = float(temp[0]), float(temp[1])


    def init_align_loss(self, loss_type):
        if loss_type == 'pearson':
            return pearson_loss
        elif loss_type == 'huber':
            return huber_loss
        elif loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif loss_type == 'MSE':
            temp = torch.nn.MSELoss(reduction='mean')
            return temp
        elif loss_type == 'kl_loss':
            return kl_loss
    def init_optimizer(self, args):
        model_name = args.model_name
        learning_rate = args.learning_rate
        learning_rate_darts = args.learning_rate_darts
        weight_decay = args.weight_decay
        model = self

        optimizer_all = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if model_name == 'NoSlct':
            optimizer_model = None
            optimizer_controller = None
        else:
            params_optimizer_model = []
            params_optimizer_darts = []
            params_optimizer_small = []

            for name, param in model.named_parameters():
                if 'controller' in name:
                    params_optimizer_darts.append(param)
                elif 'small' in name:
                    params_optimizer_small.append(param)
                else:
                    params_optimizer_model.append(param)

            optimizer_model = torch.optim.Adam(
                params=params_optimizer_model, lr=learning_rate, weight_decay=weight_decay)
            optimizer_controller = torch.optim.Adam(
                params=params_optimizer_darts, lr=learning_rate_darts, weight_decay=weight_decay)
            optimizer_small = torch.optim.Adam(
                params=params_optimizer_small, lr=learning_rate, weight_decay=weight_decay)
        return optimizer_all, optimizer_model, optimizer_controller, optimizer_small


    def pretrain(self, field, target):
        # pretrain
        res1 = self.get_predict_score_normal(
            field, emb_layer=self.emb, mlp_layer=self.mlp, BN_layer=self.BN)
        loss_normal = self.criterion(res1, target)
        res2 = self.get_predict_score_normal(
            field, emb_layer=self.emb_small, mlp_layer=self.mlp_small, BN_layer=self.BN_small,
            emb_layer_samllToNormal=None)
        loss_small = self.criterion(res2, target)

        (loss_small+loss_normal).backward()
        self.optimizer_normal.step()
        self.optimizer_small.step()

        return loss_small+loss_normal

    def forward(self, field, target, step):
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection mode for 'NAN' 'inf'

        self.zero_grad()

        if self.stage == 0:
            return self.pretrain(field, target)

        controller_position = self.two_model_controller_position
        if controller_position == 'small_emb':
            controller = self.controller_small
        elif controller_position == 'normal_emb':
            controller = self.controller
        result_small, mask, normal_field_from_small, kmax_index = self.get_predict_score(
            field, emb_layer=self.emb_small, controller=controller, mlp_layer=self.mlp_small,
        BN_layer=self.BN_small, emb_layer_samllToNormal=self.mlp_smallToNormalEmb, controller_position=controller_position)
        loss_small = self.criterion(result_small, target)

        result, normal_field = self.get_predict_score_with_mask(field, mask, kmax_index)
        loss_normal = self.criterion(result, target)

        score_align_loss = self.score_align_loss(result_small, result.detach())
        field_align_loss = self.field_align_loss((normal_field_from_small * torch.unsqueeze(mask,1)).transpose(1,2),
                                                 normal_field.detach().transpose(1,2))

        if self.small_loss_weight != 1:
            loss_small = loss_small*self.small_loss_weight
        loss = loss_normal+loss_small + \
               self.score_align_loss_w * score_align_loss + self.field_align_loss_w * field_align_loss
        # search stage, alternatively update main RS network and Darts weights
        loss.backward()
        self.optimizer_normal.step()
        self.optimizer_small.step()
        if self.stage == 1 and ((step + 1) % self.darts_frequency == 0):
            self.optimizer_controller.step()

        return loss

    def predict(self, field):
        if self.stage == 0:
            # pretrain
            result = self.get_predict_score_normal(
                field, emb_layer=self.emb, mlp_layer=self.mlp, BN_layer=self.BN)

        else:
            controller_position = self.two_model_controller_position
            if controller_position == 'small_emb':
                controller = self.controller_small
            elif controller_position == 'normal_emb':
                controller = self.controller
            result_small, mask, normal_field_from_small, kmax_index = self.get_predict_score(
                field, emb_layer=self.emb_small, controller=controller, mlp_layer=self.mlp_small,
            BN_layer=self.BN_small, emb_layer_samllToNormal=self.mlp_smallToNormalEmb, controller_position=controller_position)
            result, returned_normal_field = self.get_predict_score_with_mask(field, mask, kmax_index)
        return result


    def get_predict_score(self, field, emb_layer=None, controller=None, mlp_layer=None, BN_layer=None,
                          emb_layer_samllToNormal=None, controller_position='small_emb'):
        if (emb_layer is None) and (controller is None) and (mlp_layer is None):
            emb_layer = self.emb
            controller = self.controller
            mlp_layer = self.mlp
            BN_layer = self.BN
        field = emb_layer(field)
        if self.useBN == True:
            field = BN_layer(field)

        normal_field_from_small = emb_layer_samllToNormal(field.transpose(1, 2)).transpose(1, 2)
        if controller_position == 'normal_emb':
            weight = controller(normal_field_from_small)
        elif controller_position == 'small_emb':
            weight = controller(field)

        kmax_index, kmax_weight = kmax_pooling(weight,1,self.k)
        if self.reWeight == True:
            kmax_weight = kmax_weight/torch.sum(kmax_weight,dim=1).unsqueeze(1) # reweight, 使结果和为1
        # 创建跟weight同维度的mask，index位赋予值，其余为0
        mask = torch.zeros(weight.shape[0],weight.shape[1]).to(self.device)
        if self.useWeight:
            mask = mask.scatter_(1, kmax_index, kmax_weight) # 填充对应索引位置为weight值
        else:
            mask = mask.scatter_(1, kmax_index, torch.ones(kmax_weight.shape[0],kmax_weight.shape[1])) #对应索引位置填充1

        field = field * torch.unsqueeze(mask,1)

        res = mlp_layer(field.transpose(1,2).float())
        res = torch.sigmoid(res.squeeze(1))
        return res, mask, normal_field_from_small.clone(), kmax_index


    def get_predict_score_with_mask(self, raw_field, mask, kmax_index):
        # ******************** early selection *************************
        batch_size, seq_len = raw_field.shape
        # Step 1: Pass kmax_index into self.emb for embedding calculation
        field = self.emb(raw_field, kmax_index)  # field.shape should be [2048, embedding_dim, k]
        # Step 2: Fill the embedded result back to the original dimension according to kmax_index
        output_field = torch.zeros((batch_size, field.shape[1], seq_len), dtype=field.dtype, device=field.device)
        # Assignment using scatter_ function
        kmax_index1 = kmax_index.unsqueeze(1).expand(-1, field.shape[1], -1)
        # The values in field are dispersed into output_field, and the specific positions are specified by kmax_index.
        output_field.scatter_(2, kmax_index1, field)
        field = output_field
        # *********************************************

        if self.useBN == True:
            field = self.BN(field)

        field = field * torch.unsqueeze(mask,1)
        returned_normal_field = field.clone()

        res = self.mlp(field.transpose(1,2).float())
        res = torch.sigmoid(res.squeeze(1))
        return res, returned_normal_field

    def get_predict_score_normal(self, field, emb_layer, mlp_layer, BN_layer, emb_layer_samllToNormal=None):
        """
        不进行 feature selection
        :param field:
        :param emb_layer:
        :param mlp_layer:
        :return:
        """
        field = emb_layer(field)  # 得到相同维度 embedding
        # 对每个feature进行batchnorm
        if self.useBN == True:
            field = BN_layer(field)
        if emb_layer_samllToNormal is not None:
            field = emb_layer_samllToNormal(field.transpose(1,2)).transpose(1,2)
        res = mlp_layer(field.transpose(1,2).float())
        res = torch.sigmoid(res.squeeze(1))
        return res


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num = len(args.field_dims)
        self.embed_dim = args.embed_dim
        self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = MultiLayerPerceptron(input_dim=len(args.field_dims)*self.embed_dim,
                                        embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)
        self.BN = nn.BatchNorm1d(self.embed_dim)

    def forward(self,field):
        field = self.emb(field)
        field = self.BN(field)
        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        return torch.sigmoid(res.squeeze(1))



