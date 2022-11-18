import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn


# 构造一个pytorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 搭建模型并进行训练，多层感知机实现
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden1, n_hidden2, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.BatchNorm1d(n_hidden1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.BatchNorm1d(n_hidden2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden2, num_classes))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# 创建一个训练函数
def train_model(parser_data):
    # 读取数据
    df = pd.read_csv('./data/train.csv')
    # print(df.head())

    # 将标签和后续的28x28图像分开，由于是分类任务，所以将标签转为one-hot编码
    labels = torch.from_numpy(np.array(df['label']))

    # 将标签转为one-hot编码
    # 生成单位矩阵
    ones = torch.sparse.torch.eye(10)

    # 根据制定索引和维度保留单位矩阵中的一条数据即为one-hot编码
    label_one_hot = ones.index_select(0, labels)

    # 将训练集中的图像转换为tensor浮点类型
    df.pop('label')
    # 将训练集转为tensor
    imgs = torch.from_numpy(np.array(df))
    imgs = imgs.to(torch.float32)

    # 定义训练数据集和训练时所用的参数
    train_batch_size = parser_data.batch_size

    # 初始化学习率
    lr = parser_data.lr

    train_data = load_array((imgs, label_one_hot), batch_size=train_batch_size)

    # 使用gpu训练
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 实例化网络
    model = Net(784, 300, 100, num_classes=10)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # 记录训练过程中的train_loss和train_acc变化
    losses = []
    acces = []

    start_epoch = parser_data.start_epoch
    num_epochs = parser_data.epochs

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = 0
        train_acc = 0
        model.train()
        for img, label in train_data:
            img = img.to(device)
            label = label.to(device)
            # 计算结果
            out = model(img)
            # 计算损失
            loss = criterion(out, label)

            # 将优化器中的梯度清零
            optimizer.zero_grad()

            # 执行反向传播
            loss.backward()
            # 更新网络
            optimizer.step()

            # 记录训练loss
            train_loss += loss.item()

            # 记录训练准确度
            _, pred = out.max(1)
            l, pred1 = label.max(1)
            num_correct = (pred == pred1).sum().item()
            acc = num_correct / img.shape[0]  # 表示计算每个batch大小的acc
            train_acc += acc

        # 得到每个epoch中的loss和acc
        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))

        print('epoch:{},train_loss:{:.4f},train_acc:{:.4f}'.format(epoch,
                                                                   train_loss/len(train_data), train_acc/len(train_data)))

    print("[INFO] saving test_code results...")
    # 模型预测结果
    df_test = pd.read_csv("./data/test_code.csv")
    test_images = torch.from_numpy(np.array(df_test))
    test_images = test_images.to(torch.float32)
    test_images = test_images.to(device)
    _, pre = model(test_images).max(1)
    res = {}
    pre = pre.cpu().numpy()
    pre_size = pre.shape[0]
    num = [i for i in range(1, pre_size+1)]
    res_df = pd.DataFrame({
        'ImageId': num,
        'Label': pre
    })

    res_df.to_csv('./submission/{}_net_submit.csv'.format(parser_data.date), index=False)
    print("[INFO] Save test_code results successfully.")


# 增加参数实现对模型训练的动态调参
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data_path', default='./', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=10, type=int, help='Set the num_classes of the model.')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=1, type=int, help='Start epoch, default value is 1.')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='The number of total epochs to train.')
    # 训练的batch size
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect_ratio_group_factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    # 设置训练日期
    parser.add_argument("--date", default=True, help="Set the train date. The format is like: 202207271757")

    # 设置从checkpoint加载权重时需要设置的学习率
    parser.add_argument("--lr", default=0.01, type=float, help="The lr for a resume begin.")
    args = parser.parse_args()

    # 调用训练函数
    train_model(args)

