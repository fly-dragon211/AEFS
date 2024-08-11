import torch
import torch.nn.functional as F


class FeatureEmbedding(torch.nn.Module):
    def __init__(self, feature_num, latent_dim, initializer=torch.nn.init.xavier_uniform_):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(feature_num, latent_dim))
        initializer(self.embedding)

    def forward(self, x):
        """
        :param x: tensor of size (batch_size, num_fields)
        :return: tensor of size (batch_size, num_fields, embedding_dim)
        """
        return F.embedding(x, self.embedding)


class FeaturesLinear(torch.nn.Module):
    def __init__(self, feature_num, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(feature_num, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return : tensor of size (batch_size, 1)
        """
        return torch.sum(torch.squeeze(self.fc(x)), dim=1, keepdim=True) + self.bias


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1) if reduce_sum
                  tensor of size (batch_size, embed_dim) else
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True, use_bn=False, use_ln=False):
        """
        :param input_dim:
        :param mlp_dims: a list
        :param dropout:
        :param output_layer:
        :param use_bn:
        :param use_ln:
        """
        super().__init__()
        layers = list()
        for mlp_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, mlp_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(mlp_dim))
            if use_ln:
                layers.append(torch.nn.LayerNorm(mlp_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = mlp_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        :return : tensor of size (batch_size, mlp_dims[-1])
        """
        return self.mlp(x)


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class InnerProduct(torch.nn.Module):
    def __init__(self, field_num):
        super().__init__()
        self.rows = []
        self.cols = []
        for row in range(field_num):
            for col in range(row + 1, field_num):
                self.rows.append(row)
                self.cols.append(col)
        self.rows = torch.tensor(self.rows)
        self.cols = torch.tensor(self.cols)

    def forward(self, x):
        """
        :param x: Float tensor of size (batch_size, field_num, embedding_dim)
        :return: (batch_size, field_num*(field_num-1)/2)
        """
        batch_size = x.shape[0]
        trans_x = torch.transpose(x, 1, 2)

        self.rows = self.rows.to(trans_x.device)
        self.cols = self.cols.to(trans_x.device)

        gather_rows = torch.gather(trans_x, 2, self.rows.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans_x, 2, self.cols.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding


class __________________________:
    """
    下面的网络是基于上面的类搭建
    """
    pass

class FM(torch.nn.Module):
    def __init__(self, field_num, embed_dim):
        super().__init__()
        self.fm = FactorizationMachine(reduce_sum=True)
        self.field_num = field_num
        self.embed_dim = embed_dim

    def forward(self, x_embedding):
        """
        :param x_embedding: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1)
        """
        output_fm = self.fm(x_embedding)
        # print(output_linear.shape)
        # print(output_fm.shape)
        # logit = output_linear +  output_fm
        logit = output_fm
        return logit


class DeepFM(FM):
    def __init__(self, mlp_dims, field_num, embed_dim, dropout, use_bn=False):
        super(DeepFM, self).__init__(field_num, embed_dim)
        self.dnn_dim = field_num * embed_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, dropout, use_bn=use_bn)

    def forward(self, x_embedding):
        """
        :param x_embedding: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1)
        """
        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.reshape(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        # logit = output_dnn +  output_fm + output_linear
        logit = output_dnn + output_fm
        return logit


class DeepCrossNet(torch.nn.Module):
    def __init__(self, cross_num, mlp_dims, field_num, embed_dim, dropout, use_bn=False):
        super(DeepCrossNet, self).__init__()
        # cross_num = opt["cross"]
        # mlp_dims = opt["mlp_dims"]
        # use_bn = opt["use_bn"]
        # dropout = opt["mlp_dropout"]
        self.dnn_dim = field_num * embed_dim
        self.cross = CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x_embedding):
        """
        :param x_embedding: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1)
        """
        x_dnn = x_embedding.reshape(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit


class InnerProductNet(torch.nn.Module):
    def __init__(self, mlp_dims, field_num, embed_dim, dropout, use_bn=False):
        super(InnerProductNet, self).__init__()
        # mlp_dims = opt["mlp_dims"]
        # use_bn = opt["use_bn"]
        # dropout = opt["mlp_dropout"]
        self.field_num = field_num
        self.embed_dim = embed_dim
        self.dnn_dim = self.field_num * self.embed_dim + \
                       int(self.field_num * (self.field_num - 1) / 2)
        self.inner = InnerProduct(self.field_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)

    def forward(self, x_embedding):
        """
        :param x_embedding: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1)
        """
        x_dnn = x_embedding.reshape(-1, self.field_num * self.embed_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn = torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit

class MultiLayerPerceptronNet(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = torch.nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(torch.nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x_embedding: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1)
        """
        x = x.flatten(start_dim=1).float()
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x
