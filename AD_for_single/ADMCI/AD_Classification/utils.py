import torch
import logging
import pandas as pd
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss,model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss , model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'best_model.pth')
        logging.info(f'Validation loss decreased. Model saved.')

def gaussian_kernel(distance, sigma):
    # 计算高斯核
    return torch.exp(- (distance ** 2) / (2 * sigma ** 2))

def build_adj_matrix_similarity_add_three(high_dim_features,low_dim_features,sigma=1):
    node_num = low_dim_features.size(0)
    matrix_high = torch.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            # 计算特征向量之间的欧氏距离
            distance = torch.norm(high_dim_features[i] - high_dim_features[j], p=2)
            # 计算高斯核值
            matrix_high[i, j] = gaussian_kernel(distance, sigma)

    matrix_low = torch.zeros((node_num, node_num))
    sex = low_dim_features[:,0].unsqueeze(1)
    apoe = low_dim_features[:,4].unsqueeze(1)
    mmse = low_dim_features[:,6].unsqueeze(1)

    sex_matrix = (sex == sex.t()).float()
    apoe_matrix = (apoe == apoe.t()).float()
    mmse_diff = torch.abs(mmse - mmse.t())
    mmse_matrix = (mmse_diff <= 1).float()
    matrix_low = sex_matrix + apoe_matrix + mmse_matrix

    adj_matrix = matrix_high * matrix_low

    return adj_matrix

def build_adj_matrix_only_three(high_dim_features,low_dim_features,sigma=1):
    node_num = low_dim_features.size(0)
    
    matrix_low = torch.zeros((node_num, node_num))
    sex = low_dim_features[:,0].unsqueeze(1)
    apoe = low_dim_features[:,4].unsqueeze(1)
    mmse = low_dim_features[:,6].unsqueeze(1)

    sex_matrix = (sex == sex.t()).float()
    apoe_matrix = (apoe == apoe.t()).float()
    mmse_diff = torch.abs(mmse - mmse.t())
    mmse_matrix = (mmse_diff <= 1).float()
    adj_matrix = sex_matrix + apoe_matrix + mmse_matrix

    return adj_matrix

def build_adj_matrix_sex(high_dim_features,low_dim_features,sigma=1):
    node_num = low_dim_features.size(0)
    matrix_low = torch.zeros((node_num, node_num))
    sex = low_dim_features[:,0].unsqueeze(1)
    sex_matrix = (sex == sex.t()).float()
    adj_matrix = sex_matrix

    return adj_matrix

def build_adj_matrix_apoe(high_dim_features,low_dim_features,sigma=1):
    node_num = low_dim_features.size(0)
    matrix_low = torch.zeros((node_num, node_num))
    apoe = low_dim_features[:,4].unsqueeze(1)
    apoe_matrix = (apoe == apoe.t()).float()
    adj_matrix = apoe_matrix

    return adj_matrix

def build_adj_matrix_mmse(high_dim_features,low_dim_features,sigma=1):
    node_num = low_dim_features.size(0)
    matrix_low = torch.zeros((node_num, node_num))
    mmse = low_dim_features[:,6].unsqueeze(1)
    mmse_diff = torch.abs(mmse - mmse.t())
    mmse_matrix = (mmse_diff <= 1).float()
    adj_matrix = sex_matrix + apoe_matrix + mmse_matrix

    return adj_matrix

def extract_number(col_name):
    match = re.search(r"_(\d+)$", col_name)
    return int(match.group(1)) if match else None

def calculate_correlation_per_sample(feature_matrix, mean, var):
    # 假设 feature_matrix 的形状是 [N, 166]
    N = feature_matrix.shape[0]
    correlation_matrices = torch.empty((N, 166, 166), dtype=torch.float32)
    
    for i in range(N):
        centered_features = feature_matrix[i] - mean
        std_dev = torch.sqrt(var)
        covariance_matrix = torch.mm(centered_features.unsqueeze(0), centered_features.unsqueeze(0).T) / (166 - 1)
        correlation_matrix = covariance_matrix / torch.outer(std_dev, std_dev)
        correlation_matrices[i] = correlation_matrix.squeeze()  # 确保移除了不必要的单一维度
    
    return correlation_matrices

def cov_builder(high_dim_features,label):
    labels_CN = label == 0
    
    # 定义每组特征的起始和结束索引
    c1_start, c1_end = 0, 166
    c2_start, c2_end = 166, 332
    c3_start, c3_end = 332, 498
    
    # 选择对应组的特征，并计算每组特征的均值和方差
    c1_mean = high_dim_features[labels_CN, c1_start:c1_end].mean(dim=0)
    c1_var = high_dim_features[labels_CN, c1_start:c1_end].var(dim=0)
    
    c2_mean = high_dim_features[labels_CN, c2_start:c2_end].mean(dim=0)
    c2_var = high_dim_features[labels_CN, c2_start:c2_end].var(dim=0)
    
    c3_mean = high_dim_features[labels_CN, c3_start:c3_end].mean(dim=0)
    c3_var = high_dim_features[labels_CN, c3_start:c3_end].var(dim=0)

    # 计算相关性矩阵
    correlation_matrices_c1 = calculate_correlation_per_sample(high_dim_features[:, c1_start:c1_end], c1_mean, c1_var)
    correlation_matrices_c2 = calculate_correlation_per_sample(high_dim_features[:, c2_start:c2_end], c2_mean, c2_var)
    correlation_matrices_c3 = calculate_correlation_per_sample(high_dim_features[:, c3_start:c3_end], c3_mean, c3_var)
    # 堆叠相关性矩阵
    correlations_stack = torch.stack((correlation_matrices_c1, correlation_matrices_c2, correlation_matrices_c3), dim=1)
    correlations_stack[torch.isnan(correlations_stack)] = 0
    return correlations_stack

def flatten_matrix(correlation_matrices):
    flattened_vectors = []
    rows, cols = torch.triu_indices(166, 166, offset=1)
    for matrix in correlation_matrices:
        # 使用上三角的索引来选择元素，并将其拉平
        flattened_vector = matrix[rows, cols]
        # 将拉平后的向量添加到列表中
        flattened_vectors.append(flattened_vector)
    flattened_vectors_tensor = torch.stack(flattened_vectors)
    return flattened_vectors_tensor

def individual_feature_builder(high_dim_features,label):
    labels_CN = label == 0
    
    # 定义每组特征的起始和结束索引
    c1_start, c1_end = 0, 166
    c2_start, c2_end = 166, 332
    c3_start, c3_end = 332, 498
    
    # 选择对应组的特征，并计算每组特征的均值和方差
    c1_mean = high_dim_features[labels_CN, c1_start:c1_end].mean(dim=0)
    c1_var = high_dim_features[labels_CN, c1_start:c1_end].var(dim=0)
    
    c2_mean = high_dim_features[labels_CN, c2_start:c2_end].mean(dim=0)
    c2_var = high_dim_features[labels_CN, c2_start:c2_end].var(dim=0)
    
    c3_mean = high_dim_features[labels_CN, c3_start:c3_end].mean(dim=0)
    c3_var = high_dim_features[labels_CN, c3_start:c3_end].var(dim=0)

    # 计算相关性矩阵
    correlation_matrices_c1 = calculate_correlation_per_sample(high_dim_features[:, c1_start:c1_end], c1_mean, c1_var)
    correlation_matrices_c2 = calculate_correlation_per_sample(high_dim_features[:, c2_start:c2_end], c2_mean, c2_var)
    correlation_matrices_c3 = calculate_correlation_per_sample(high_dim_features[:, c3_start:c3_end], c3_mean, c3_var)
    
    # 拉平后的张量
    flatten_matrix_c1 = flatten_matrix(correlation_matrices_c1)
    flatten_matrix_c2 = flatten_matrix(correlation_matrices_c2)
    flatten_matrix_c3 = flatten_matrix(correlation_matrices_c3)
    
    # 合并成一个长向量
    combined_tensor = torch.cat((flatten_matrix_c1, flatten_matrix_c2, flatten_matrix_c3), dim=1)
    combined_tensor[torch.isnan(combined_tensor)] = 0
    return combined_tensor

# 读取脑区与脑区之间的邻接矩阵
def read_brain_region_adjacency(brain_region_adjacency_path):
    brain_region_adjacency = pd.read_csv(brain_region_adjacency_path)
    brain_region_adjacency_np = brain_region_adjacency.values
    brain_region_adjacency_tensor = torch.tensor(brain_region_adjacency_np)
    return brain_region_adjacency_tensor
    
    