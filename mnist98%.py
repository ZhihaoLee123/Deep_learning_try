import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 定义超参数
batch_size = 512
learning_rate_1 = 0.01
epochs = 10

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    # TODO: 添加数据转换，例如将数据转换为张量，并进行归一化
])

# TODO: 1. 加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset= datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=36, shuffle=False)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5,stride=1,padding=0)
        self.conv2 = nn.Conv2d(10, 20, 3, stride=1, padding=0)
        self.pool1=nn.MaxPool2d(2,2,0)

        self.fc1=nn.Linear(2000,500)
        self.fc2 = nn.Linear(500, 10)

        # TODO: 2. 定义可学习的权重参数，例如全连接层


    def forward(self, x):
        in_size=x.size(0)
        out=self.conv1(x)
        out=nn.functional.relu(out)
        out=self.pool1(out)
        out=self.conv2(out)
        out=nn.functional.relu(out)
        out=out.view(in_size,-1)
        out=self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out=nn.functional.log_softmax(out,1)
        return out

        # TODO: 3. 实现前向传播过程



# 初始化模型
model = SimpleNN()

# TODO: 4. 定义损失函数，例如使用交叉熵损失
criterion = nn.NLLLoss()  # 替换为合适的损失函数

# TODO: 5. 定义优化器，例如使用 SGD 或 Adam
optimizer = optim.Adam(model.parameters(),lr=learning_rate_1)  # 替换为合适的优化器


def calculate_accuracy(loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，节省内存和提高速度
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测的标签
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

    accuracy = 100 * correct / total  # 计算准确率
    return accuracy



# 训练网络
for epoch in range(epochs):
    for images, labels in train_loader:
        # 6. 清空梯度
        optimizer.zero_grad()

        # 7. 处理输入并获取输出
        outputs = model(images)  # 替换为前向传播的结果

        # 8. 计算损失
        loss = criterion(outputs,labels)  # 替换为损失计算

        # 9. 反向传播
        loss.backward()  # GPU加速可使用 .cuda() 转移到GPU

        # 10. 更新网络参数
        optimizer.step()

    # 打印每个 epoch 的损失
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    print(f"accuracy:{calculate_accuracy(test_loader)}")
print("Training complete.")
