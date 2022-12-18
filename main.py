import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch_geometric.nn import GCNConv
# 生成图形数据
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

# 定义 GCN 模型
class GCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(GCN, self).__init__()
        # self.conv1 = nn.Linear(num_features, 16)
        # self.conv2 = nn.Linear(16, num_classes)
        self.conv1 = GCNConv(num_features,16)
        self.conv2 = GCNConv(16,num_classes)

    def forward(self, x, adj):
        # x = self.conv1(x)
        # x = torch.relu(x)
        # x = self.conv2(x)
        x = self.conv1(x,adj)
        x = self.conv2(x,adj)
        return x

# 初始化模型
model = GCN(num_nodes=len(G), num_features=2, num_classes=2)

# 准备训练数据
adj = np.array([[0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0]])
x = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])
y = np.array([0, 0, 0, 1])

train_data = list(zip(x, adj, y))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 开始训练
for epoch in range(100):
    for x, adj, y in train_loader:
        x=x.to(torch.float32)
        adj=adj.to(torch.float32)
        y=y.type(torch.LongTensor)
        optimizer.zero_grad()
        output = model(x, adj)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# 准备测试数据
adj_test = np.array([[0, 1, 1, 0],
                     [1, 0, 1, 0],
                     [1, 1, 0, 1],
                     [0, 0, 1, 0]])
x_test = np.array([[5, 6],
                   [6, 7],
                   [7, 8],
                   [8, 9]])
y_test = np.array([1, 0, 0, 1])

test_data = list(zip(x_test, adj_test, y_test))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

# 使用训练好的 GCN 模型进行预测
with torch.no_grad():
    prediction = []
    for x, adj, y in test_loader:
        x=x.to(torch.float32)
        adj=adj.to(torch.float32)
        output = model(x, adj)
        prediction.extend(output.max(1)[1].tolist())

# 计算准确率、召回率和 F1 值
accuracy = accuracy_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)

# 输出结果
print("Accuracy: {:.2f}".format(accuracy))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

