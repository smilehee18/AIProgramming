import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

cost_history = []
accuracy_history = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device : ", device)

#normalization
transforms_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize((0.4914), (0.2470))])
transforms_test = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize((0.4914), (0.2470))])

#dataset setting
mnist_train = dsets.MNIST(root = 'MNIST_data/',
                          train = True,
                          transform = transforms_train,
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms_test,
                         download=True)
#dataset loader
trainloader = DataLoader(dataset=mnist_train, batch_size=128, shuffle=True, drop_last=True)
testloader = DataLoader(mnist_test, batch_size=128, shuffle=False, drop_last=True)

#mnist image data shape 28 x 28 = 784
#model
class Network(nn.Module):
    def __init__(self): #생성자 파라미터 초기화 및 함수 선언부
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), (1, 1))
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc_code = nn.Linear(in_features=64, out_features=64)
        self.fc_output = nn.Linear(64, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.55)
        self.bn = nn.BatchNorm2d(64, momentum=0.5)
    def forward(self, x):
        feature_map = self.conv1(x)
        activated = self.relu(feature_map)
        pooled = self.maxpool(activated)
        x = self.maxpool(self.relu(self.conv2(pooled)))
        x = self.bn(self.maxpool(self.relu(self.conv3(x))))
        x = x.view(x.size(0),-1) #128개의 feature들이 일렬로 나열
        code = self.dropout(self.relu(self.fc_code(x))) #128 -> 128
        output = self.fc_output(code)
        return output

#model 객체 생성 및 함수 선언
model = Network()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01) #Adam 경사하강법
criterion = nn.CrossEntropyLoss(reduction='mean')  #손실 함수

#train
for epoch in range(20):
    model.train()
    total_batch = len(trainloader)
    for idx, (X, Y) in enumerate(trainloader):
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device)
        optimizer.zero_grad()

        outputs = model(X)
        cost = criterion(outputs, Y)
        cost.backward()
        optimizer.step()

        correct_pred = torch.argmax(outputs, 1) == Y
        accur = (correct_pred.sum().item()/len(correct_pred))
        accuracy_history.append(accur)
        cost_history.append(cost.item())
        print(idx)
    print('Epoch {:d} COST: {:.5f} Accuracy: {:.6f}%'.format(epoch, cost.item(), accur*100))

plt.rcParams['axes.grid'] = True
fig, axes = plt.subplots(2, 1)
axes[0].plot(cost_history, 'r--')
axes[0].set_title('cost')
axes[1].plot(accuracy_history, 'b--')
axes[1].set_title('accuracy')
plt.tight_layout()
plt.show()
print('Learning Finished!')

#TEST
model.eval()
with torch.no_grad():
    #정확도 측정
    for x_test, y_test in testloader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        yhat = model(x_test)
        #print(y_test.size()) #128
        #print(yhat.size())   #128 x 10 (batchsize = 128)
        pred = torch.argmax(yhat, 1) == y_test
        accuracy = pred.float().mean()
        print('Accuracy : ', accuracy.item()) #0.9

    r = random.randint(0, len(mnist_test) -1)
    x_single_data = mnist_test.test_data[r:r+1].float().to(device)
    y_single_data = mnist_test.test_labels[r:r+1].to(device)
    x_single_data = x_single_data.unsqueeze(dim=0) #[1, 28, 28] -> [1, 1, 28, 28]

    #shape 문제 존재 -> 위 코드로 해결
    print('Label : ',y_single_data.item()) #item 하는 이유?
    single_pred = model(x_single_data)
    print('pred : ', torch.argmax(single_pred, 1).item())

    #GPU-> CPU로 메모리 복사 후 시각화
    x_single_data = x_single_data.squeeze().cpu()
    plt.imshow(x_single_data, cmap='Greys', interpolation='nearest')
    plt.grid(False)
    plt.show()