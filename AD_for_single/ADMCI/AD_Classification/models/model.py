# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, ChebConv

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        return self.bn(F.relu(self.linear(x)))
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)
        attention = torch.softmax(Q @ K.transpose(-2, -1) / (self.embed_size ** 0.5), dim=-1)
        out = attention @ V
        return out

# 图表示学习的准备函数
class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
    
# 图表示学习的主函数
class Graph_based_GCN(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels,num_features):
        super(Graph_based_GCN, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.graph_network = GraphNeuralNetwork(num_features, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels + embedding_dim, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.classfier = nn.Linear(hidden_channels, output_dim)
        

    def forward(self, high_dim_features,low_dim_features,brain_edge_index,edge_index):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features_reshaped = high_dim_features.view(high_dim_features.shape[0], 166, 3)
        graph_features = self.graph_network(high_dim_features_reshaped, brain_edge_index)
        # 均值全局化
        global_graph_feature = graph_features.mean(dim=1)  # 假设使用平均值聚合
        # concat全局化
        # global_graph_feature = graph_features.view(graph_features.shape[0], -1)
        # print(global_graph_feature.shape)
        # 将表型特征与全图特征结合
        combined_features = torch.cat((global_graph_feature, low_dim_embedded), dim=1)
        # 使用全连接层进行分类
        x = F.relu(self.fc1(combined_features))
        
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p = 0.3)
        
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
    
# 图表示学习的主函数
class Graph_based_MLP(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels,num_features):
        super(Graph_based_MLP, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.graph_network = GraphNeuralNetwork(num_features, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels + embedding_dim, embedding_dim)
        self.classfier = nn.Linear(embedding_dim, output_dim)

    def forward(self, high_dim_features,low_dim_features,brain_edge_index):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features_reshaped = high_dim_features.view(high_dim_features.shape[0], 166, 3)
        graph_features = self.graph_network(high_dim_features_reshaped, brain_edge_index)
        # 均值全局化
        global_graph_feature = graph_features.mean(dim=1)  # 假设使用平均值聚合
        # concat全局化
        # global_graph_feature = graph_features.view(graph_features.shape[0], -1)
        # print(global_graph_feature.shape)
        # 将表型特征与全图特征结合
        combined_features = torch.cat((global_graph_feature, low_dim_embedded), dim=1)
        # 使用全连接层进行分类
        x = F.relu(self.fc1(combined_features))
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x

# 经过变换的高维特征
class test_high_correlation(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(test_high_correlation, self).__init__()
        self.bn_low = nn.BatchNorm1d(17)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        self.bn_embedding1 = nn.BatchNorm1d(embedding_dim*3)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.linear_embedding1 = MLP(embedding_dim*3, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.self_attention = SelfAttention(embedding_dim)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classfier = nn.Linear(embedding_dim, output_dim)
        self.linear_conv = MLP(82668, embedding_dim)
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)  # 举例使用3x3的卷积核

    def forward(self, high_dim_features, low_dim_features,high_dim_cov_matrix):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)

        high_dim_cov_matrix = high_dim_cov_matrix  # 增加channel维度
        conv_output = self.conv2d(high_dim_cov_matrix)
        conv_output = conv_output.flatten(1)  # 展平除了batch维度的所有维度
        conv_features = self.linear_conv(conv_output)
        
        # 将卷积层的输出与高维特征连接起来
        combined_features = torch.cat((combined_features, conv_features), dim=1)
    
        x = self.linear_embedding1(self.bn_embedding1(combined_features))
        x = F.relu(x)
        x = F.dropout(x, p = 0.3)
        
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
    
# 图卷积神经网络 + 高、低维特征嵌入
class Baseline_GCN(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(Baseline_GCN, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        
        self.conv1 = GCNConv(embedding_dim * 2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classfier = nn.Linear(hidden_channels, output_dim)

    def forward(self, high_dim_features, low_dim_features, edge_index):
        
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        
        x = self.conv1(combined_features, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p = 0.4)
        
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, p = 0.1)
        
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
    
class Adjacency_based_GCN(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(Adjacency_based_GCN, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.conv1 = GCNConv(embedding_dim * 2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classfier = nn.Linear(hidden_channels, output_dim)

    def forward(self, high_dim_features, low_dim_features, edge_index,high_dim_cov_matrix_flatten):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_cov_matrix_flatten))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)

        x = self.conv1(combined_features, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p = 0.5)
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
    
class Adjacency_2D_based_GCN(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(Adjacency_2D_based_GCN, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.conv1 = GCNConv(embedding_dim * 3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classfier = nn.Linear(hidden_channels, output_dim)
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)  # 举例使用3x3的卷积核
        self.linear_conv = MLP(82668, embedding_dim)

    def forward(self, high_dim_features, low_dim_features, edge_index,high_dim_cov_matrix):
        
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        high_dim_cov_matrix = high_dim_cov_matrix  # 增加channel维度
        conv_output = self.conv2d(high_dim_cov_matrix).flatten(1)  # 展平除了batch维度的所有维度
        conv_features = self.linear_conv(conv_output)
        
        # 将卷积层的输出与高维特征连接起来
        combined_features = torch.cat((combined_features, conv_features), dim=1)
        
        x = self.conv1(combined_features, edge_index)
        x = F.tanh(x)
        x = F.dropout(x, p = 0.5)
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
            
# 只用高维的GCN
class Only_high_GCN(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(Only_high_GCN, self).__init__()
        #self.bn_low = nn.BatchNorm1d(17)
        #self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classfier = nn.Linear(hidden_channels, output_dim)

    def forward(self, high_dim_features, low_dim_features, edge_index):
        
        #low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        #combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        x = self.conv1(high_dim_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.1)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p = 0.1)
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x

# 2D卷积-邻接矩阵
class Adjacency_2D_based_MLP(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(Adjacency_2D_based_MLP, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.bn_embedding1 = nn.BatchNorm1d(embedding_dim*3)
        self.linear_embedding1 = MLP(embedding_dim*3, embedding_dim)
        self.classfier = nn.Linear(embedding_dim, output_dim)
        self.linear_conv = MLP(82668, embedding_dim)
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1)  # 举例使用3x3的卷积核
        
    def forward(self, high_dim_features, low_dim_features, high_dim_cov_matrix):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        high_dim_cov_matrix = high_dim_cov_matrix  # 增加channel维度
        conv_output = self.conv2d(high_dim_cov_matrix).flatten(1)  # 展平除了batch维度的所有维度
        conv_features = self.linear_conv(conv_output)
        
        # 将卷积层的输出与高维特征连接起来
        combined_features = torch.cat((combined_features, conv_features), dim=1)
        x = self.linear_embedding1(self.bn_embedding1(combined_features))
        x = F.relu(x)
        x = F.dropout(x, p = 0.3)
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
    
# 邻接矩阵拉平加入
class Adjacency_based_MLP(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(Adjacency_based_MLP, self).__init__()
        self.bn_low = nn.BatchNorm1d(17)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.mlp1 = nn.Linear(embedding_dim*2, embedding_dim*2)
        self.mlp2= nn.Linear(embedding_dim*2, embedding_dim*2)
        self.classfier = nn.Linear(embedding_dim*2, output_dim)
        
    def forward(self, high_dim_features, low_dim_features, high_dim_cov_matrix_flatten):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_cov_matrix_flatten))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        x = combined_features
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x

# 高低维特征嵌入 + MLP
class Baseline_MLP(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(Baseline_MLP, self).__init__()
        self.bn_low = nn.BatchNorm1d(low_dim_input_size)
        self.linear_low = MLP(low_dim_input_size, embedding_dim)
        self.bn_high = nn.BatchNorm1d(high_dim_input_size)
        self.linear_high = MLP(high_dim_input_size, embedding_dim)
        self.classfier = nn.Linear(embedding_dim*2, output_dim)
        
    def forward(self, high_dim_features, low_dim_features, edge_index):
        low_dim_embedded = self.linear_low(self.bn_low(low_dim_features))
        high_dim_features = self.linear_high(self.bn_high(high_dim_features))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=-1)
        x = combined_features
        x = self.classfier(x)
        x = F.softmax(x, dim=1)
        x = torch.log(x)
        return x
    
'''                         以下暂时没用                            '''    
# 图卷积神经网络
class GCN(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels=128):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dim_nodes, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

# 切比雪夫-图卷积神经网络
class Cheb(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels):
        super(Cheb, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = ChebConv(dim_nodes, hidden_channels, K=3)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=3)
        self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, *adjacency):
        x = self.conv1(x, *adjacency)
        x = F.relu(x)
        x = self.conv2(x, *adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

# 图注意力网络
class GAT(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels=8, heads=8, output_channels=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_nodes, hidden_channels, heads=heads, dropout=0.6)
        # On the last layer we reduce the output heads to 1, meaning we concatenate
        # the output of the previous heads from 8*hidden_channels to hidden_channels again.
        self.conv2 = GATConv(hidden_channels*heads, output_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, adjacency):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, adjacency))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, adjacency)
        return F.log_softmax(x, dim=1)
    

# 手动写卷积操作，保留在这但是可能用不上    
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

class GcnNet(nn.Module):
    def __init__(self, input_dim, hidden_channels=64):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_channels)
        self.gcn2 = GraphConvolution(hidden_channels, 2)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


