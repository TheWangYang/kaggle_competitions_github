import torch
from torch import nn
from .backbone import AlexNet
import numpy as np


# 创建模型函数
def create_model(parser_data):
    model = AlexNet(num_classes=parser_data.num_classes)
    return model


@torch.no_grad()
def evaluate_model(model, val_loader, criterion):
    # 将模型设置为评价模式
    model.eval()
    # 设置curr_loss
    curr_loss = 0
    # 设置curr_acc
    curr_acc = 0
    for data in val_loader:
        images, labels = data

        # batch_size, dim, width, height
        images = Variable(images.view(1, 1, 28, 28))
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        # 记录loss变化
        curr_loss += loss.item()

        # 记录训练准确度
        value, pred = torch.max(outputs, 1)
        num_right = (pred == labels).sum().item()
        acc = num_right / images.shape[0]
        curr_acc += acc
    return curr_loss / len(val_loader), curr_acc / len(val_loader)


# 创建一个训练函数
def train_model(parser_data):

    # 训练数据预处理
    # 导入数据
    train_data = pd.read_csv('./data/train.csv', dtype=np.float32)
    test_data = pd.read_csv('./data/test_code.csv', dtype=np.float32)

    # 处理数据
    train_np = train_data.values
    x_test = test_data.values
    y_train_np = train_np[:, 0]  # 表示得到index=0列对应的label
    x_train_np = train_np[:, 1:]  # 表示得到1到最后一列的28x28灰度图像对应的像素值

    # 数据分割
    x_train_np, x_valid_np, y_train_np, y_valid_np = \
        train_test_split(x_train_np, y_train_np, test_size=0.2, random_state=42)
    x_train_ts = torch.from_numpy(x_train_np)
    y_train_ts = torch.from_numpy(y_train_np).type(torch.LongTensor)

    # 得到batch_size大小
    batch_size = parser_data.batch_size

    # 设置数据加载配置
    train_dataset = torch.utils.data.TensorDataset(x_train_ts, y_train_ts)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True)

    # 创建验证集
    x_val_ts = torch.from_numpy(x_valid_np)
    y_val_ts = torch.from_numpy(y_valid_np).type(torch.LongTensor)

    val_dataset = torch.utils.data.TensorDataset(x_val_ts, y_val_ts)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    print("[INFO] data preparation finished...")

    # 实例AlexNet，初始化模型时，设置num_classes = 10，为数据集中的类别总数
    model = create_model(parser_data)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # 设置使用动量损失momentum，和权重减少weight_decay
    # optimizer = torch.optim.SGD(model.parameters(), lr=parser_data.lr)
    # 设置Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=parser_data.lr)

    # betas=(0.9, 0.999), eps=1e-08,
    #                                  weight_decay=0

    # 设置动态学习率计划调整
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    # 设置train_loss和train_acc数组保存每个epoch过后的train_loss和train_acc
    train_loss = []
    train_acc = []

    # 得到开始训练epoch
    start_epoch = parser_data.start_epoch
    # 得到训练总的epoch
    num_epochs = parser_data.epochs

    print("[INFO] train model start...")
    # 训练过程
    for epoch in range(start_epoch, num_epochs + 1):
        # 在每个epoch结束之后，将模型更改为train模式
        model.train()
        # 设置记录训练过程中的train_loss和train_acc
        curr_loss = 0
        curr_acc = 0
        # 对于每个epoch中的多个 batch_size进行训练
        for data in train_loader:
            images, labels = data

            images = Variable(images.view(batch_size, 1, 28, 28))
            labels = Variable(labels)

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录loss变化
            curr_loss += loss.item()

            # 记录训练准确度
            value, pred = torch.max(outputs, 1)
            num_right = (pred == labels).sum().item()
            acc = num_right / images.shape[0]
            curr_acc += acc

        # 在每个epoch结束之后，调用学习率计划
        lr_scheduler.step()

        # 保存每个epoch中的loss和acc
        train_loss.append(curr_loss / len(train_loader))
        train_acc.append(curr_acc / len(train_loader))

        # 打印当前epoch训练之后的模型的loss和acc情况
        print('epoch:{},train_loss:{:.4f},train_acc:{:.4f}'.format(epoch,
                                                                   curr_loss / len(train_loader),
                                                                   curr_acc / len(train_loader)))

        print("[INFO] Evaluating current model on the validation dataset...")
        # 在每个epoch之后计算当前模型在验证集上的val_loss和val_acc
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        # 打印val_loss和val_acc
        print('epoch:{},val_loss:{:.4f},val_acc:{:.4f}'.format(epoch, val_loss, val_acc))

    # 将model放到cpu中
    model = model.cpu()

    print("[INFO] saving test_code results...")
    # 输出到文件
    test_results = np.zeros((x_test.shape[0], 2), dtype='int32')  # test_results.size=(测试样本个数，2)
    for i in range(x_test.shape[0]):  # 对于每一个测试样本
        one_image = torch.from_numpy(x_test[i]).view(1, 1, 28, 28)  # one_image=第i个测试样本，shape=1*1*28*28
        one_output = model(one_image)  # 将one_image输入到模型中
        test_results[i, 0] = i+1  # test_results第0列表示第i个样本（从1到m）
        test_results[i, 1] = torch.max(one_output.data, 1)[1].numpy()  # 第一列表示预测的类别

    # 将上述数据添加到csv文件中
    Data = {'ImageId': test_results[:, 0], 'Label': test_results[:, 1]}
    DataFrame = pd.DataFrame(Data)
    save_date = parser_data.date
    DataFrame.to_csv('./submission/{}_alexnet_submit.csv'.format(save_date), index=False, sep=',')

    print("[INFO] save test_code results finished...")


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




