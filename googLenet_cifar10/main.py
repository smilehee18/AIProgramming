import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from googlenet import GoogLeNet
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random

cost_history = []
accuracy_history = []
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_tensor(tensor, num_col=None, label=None):
        tensor = tensor.detach()
        tensor = tensor.permute(0, 2, 3, 1)     # (32, 3, 7, 7) -> (32, 7, 7, 3) 차원을 바꾼다
        npArr = tensor.cpu().numpy()            # 텐서를 cpu에서 numpy로 변환
        num_row = int(npArr.shape[0] / num_col) # 행의 개수가 몇 개인지? shape[0] : 전체크기 / col(행의 크기)
        fig, ax = plt.subplots(num_row, num_col)
        idx = 0                                 # idx 0 초기화
        for r in range(num_row):                # num_row만큼 반복
            for c in range(num_col):
                ax[r, c].imshow(npArr[idx,])    # 이미지 보여주기
                ax[r, c].set_xticks([])         # x축 설정하기
                ax[r, c].set_yticks([])         # y축 설정하기
                if label is not None:
                    ax[r, c].set_title('label: {}\npred: {}'.
                                       format(label[0][idx], label[1][idx]), fontsize=5) #라벨, pred 제목 설정하기
                idx+=1  #반복문 돌때마다 인덱스 증가
        plt.tight_layout()
        plt.show()


if not os.path.isdir("./checkpoint"):
    os.makedirs("./checkpoint")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)

transforms_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4464), (0.2470, 0.2435, 0.2616))])
transforms_test = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4464), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, drop_last=False)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, drop_last=False)

model = GoogLeNet()
model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), momentum=0.5, lr=0.01)
criterion = nn.CrossEntropyLoss(reduction='mean')

for epoch in range(10):
    model.train()
    for idx, (X, Y) in enumerate(trainloader):
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device)
        optimizer.zero_grad()

        outputs, _, _ = model(X) #torch.Size([512, 10])
        loss = criterion(outputs, Y)

        loss.backward()
        optimizer.step()

        correct_pred = torch.argmax(outputs, 1) == Y
        accur = (correct_pred.sum().item() / len(correct_pred))
        accuracy_history.append(accur)
        cost_history.append(loss.item())
        #print(idx)
    print('Epoch {:d} COST: {:.5f} Accuracy: {:.6f}%'.format(epoch, loss.item(), accur * 100))

    path = 'checkpoint/model_state_%d.st'%(epoch)
    torch.save(model.state_dict(),path)

plt.rcParams['axes.grid'] = True
fig, axes = plt.subplots(2, 1)
axes[0].plot(cost_history, 'r--')
axes[0].set_title('cost')
axes[1].plot(accuracy_history, 'b--')
axes[1].set_title('accuracy')
plt.tight_layout()
plt.show()
print('Learning Finished!')

model.eval()
with torch.no_grad():
    for x_test, y_test in testloader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        model.aux_logits = False
        yhat = model(x_test)
        # print(y_test.size()) #256
        # print(yhat.size())   #256 x 10 (batchsize = 256)
        pred = torch.argmax(yhat, 1) == y_test
        accuracy = pred.float().mean()
        print('Accuracy : ', accuracy.item())
        _, pred2 = yhat.topk(1, 1, largest=True, sorted=True)
        correct_list = [classes[x] for x in y_test.cpu().numpy()]
        pred_list = [classes[x] for x in pred2.cpu().squeeze().numpy()]
        print(correct_list)
        print(pred_list)
        #plot_tensor(x_test, num_col=4, label=[correct_list, pred_list])
        break

    r = random.randint(0, len(testset) -1) #랜덤으로 1개 빼오기
    x_single_data = torch.Tensor(testset.data) #shape이 뭔지?
    x_single_data = x_single_data.type(dtype=torch.FloatTensor)
    x_single_data = x_single_data[r:r+1,:,:,:] #1개 이미지만 빼오기
    x_single_data = x_single_data.permute(0,3,1,2) #차원 재배치 (1, 32, 32, 3) -> (1, 3, 32, 32)

    y_single_data = testset.targets[r:r+1]
    y_single_data = torch.Tensor(y_single_data)

    single_pred = model(x_single_data.to(device))
    print('Label : ', classes[int(y_single_data.item())])  # item 하는 이유 -> 진짜값만 뽑아올려고
    print(y_single_data) #인덱스 값
    print(y_single_data.item())
    print('pred : ', classes[torch.argmax(single_pred, 1).item()])

    # dim=0 차원 축소, plt로 그리기 위해 재배치, numpy로 변환
    x_single_data = x_single_data.squeeze().permute(1, 2, 0).numpy() / 255.0
    # (1, 3, 32, 32) -> (3, 32, 32) -> (32, 32, 3)
    plt.imshow(x_single_data)
    plt.grid(False)
    plt.show()
