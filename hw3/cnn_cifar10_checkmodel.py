import torch
import torch.nn as nn                        #nn모듈 import
import torchvision
import torchvision.transforms as transforms  #데이터 증강을 위한 lib
import numpy as np
import matplotlib.pyplot as plt              #그림 그리는 lib

#정확도를 측정하는 함수
def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        num_cor = []
        for k in topk: #위에서부터 1등까지 반복하겠다
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            num_cor.append(correct_k.clone())
            acc.append(correct_k.mul(1/batch_size))
    return acc, num_cor


#가중치를 초기화하는 함수
def init_weights(m, init_type='normal', init_gain=0.02):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): #속성값 비교
        if init_type == 'normal':        #타입이 normal이라면
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':       #타입이 xavier이라면
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':   #직교하는 성질 이용
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

#그래프나 그림 그리는 함수
def plot_tensor(tensor, mode=1, num_col=None, label=None):
    if mode == 1:                               # 모드가 1이라면
        tensor = tensor.detach()                # 그림 그릴때는 텐서에서 분리한다
        tensor = tensor.permute(0, 2, 3, 1)     # (32, 3, 7, 7) -> (32, 7, 7, 3) 차원을 아예 바꾼다
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
    if mode == 2:                        #모드가 2라면
        tensor = tensor.detach().cpu()   # detach from tensor graph
        img_grid = torchvision.utils.make_grid(tensor, nrow=num_col, padding=2, pad_value=1, normalize=True)
        plt.imshow(img_grid.permute((1, 2, 0)))  #(w,h,c) 순서로 재배치하여 이미지 보이기
        plt.axis('off')                          #축은 설정하지 않을게요
        plt.tight_layout()
        plt.show()

#특정 feature 맵을 확인하기 위한 클래스 정의
class LayerActivations:
    features = []
    def __init__(self, model, layer_num):
        pass
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.detach().numpy()

    def remove(self):
        self.hook.remove()

# CustomLayer 구성
class Network(nn.Module):
    def __init__(self): #생성자 함수와 같은 역할
        super(Network, self).__init__() #부모 메소드 상속 받고
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7,7), stride=(1,1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), (1,1)) #입력 채널 32, 커널 갯수 64, 커널 사이즈 3x3
        self.conv3 = nn.Conv2d(64, 128, 3, 1)        #입력 채널 64, 커널 갯수 128, 커널 사이즈 3x3
        self.fc_code = nn.Linear(in_features=128, out_features=128)  #입력 특징 128, 출력도 128
        self.fc_output = nn.Linear(128, 10)          #입력 128, 출력 10 -> 총 10개의 클래스로 분류
        self.relu = nn.ReLU(inplace=True)            #reLu activation 메소드를 이용하여 값자체를 바꿀게
        self.maxpool = nn.MaxPool2d(2)               #maxPool 방식을 이용하여 feature map의 크기를 줄임
        self.dropout = nn.Dropout(p=0.5)             #dropout응 이용하여 overfitting 방지

    def forward(self, x):                             # layer의 순서 부분
        feature_map = self.conv1(x)                   #convolution 연산을 한다
        activated = self.relu(feature_map)            #활성화 함수를 씌운다
        compressed = self.maxpool(activated)          #pooling을 통해 feature map의 크기를 감소
        x = self.maxpool(self.relu(self.conv2(compressed)))  #한번에 연산한것
        x = self.maxpool(self.relu(self.conv3(x)))         #conv3을 통해 한번에 연산한 것
        x = x.view(x.size(0), -1)                          #view로 사이즈를 바꾸어서 일자로 쭉 핀다
        code = self.dropout(self.relu(self.fc_code(x)))    #dropout 계산하기
        output= self.fc_output(code)                       #128개 -> 10개 확률
        return output, code                                #확률과 128개의 특징값 추출

    def visualize_conv1(self, map):
        feature_map = self.conv1(map) #conv1 객체를 가지고
        activated = self.relu(feature_map) #relu 활성화함수 적용
        return activated


if __name__ == '__main__':  #메인함수
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #디바이스 정하기
    print(device) #연산장치 프린트하기

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    transform_test = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    model = Network() #CustomLayer 객체 생성 및 선언
    model.to(device) #model를 디바이스 위에 올린다
    state_dict = torch.load('./ckpt_0/model_state_61.st') #61번째 모델을 로드할게요
    model.load_state_dict(state_dict, strict=True)       #model에 저장하기

    weight1_tensor = model.conv1.weight                  #(32, 3, 7, 7), 첫번째 커널(필터) 가중치값
    #conv1_tensor = model.conv1.forward(weight1_tensor)  #32, 32, 1, 1 -> 1x1 feature map이 32개
    weight2_tensor = model.conv2.weight                  #(64, 32, 3, 3)
    weight3_tensor = model.conv3.weight                  #(64, 128, 3, 3)
    weight4_tensor = model.relu
    weight5_tensor = model.fc_code.weight
    weight6_tensor = model.fc_output.weight
    weight7_tensor = model.maxpool
    weight8_tensor = model.dropout

    plot_tensor(weight1_tensor, mode=2, num_col=4)  #plot_tensor를 호출함으로써 가시화  #(32, 3, 7, 7)
    #plot_tensor(weight2_tensor, mode=1, num_col=4)
    #plot_tensor(weight3_tensor, mode=1, num_col=4)
    #plot_tensor(conv3_tensor, mode=1, num_col=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #10개의 클래스 정의
    model.eval() #모델을 Test용도로 쓸게요
    with torch.no_grad(): #gradient 적용 X
        for step, (data, targets) in enumerate(testloader): #testloader만큼 반복하면서
            data = data.to(device, dtype=torch.float) #data를 디바이스 위에 올린다
            output, _ = model(data)                   #model(data)의 리턴값을 저장
            _, pred = output.topk(1, 1, True, True)   #topk로 줄 세우기
            target_cls = [classes[x] for x in targets.cpu().numpy()] #라벨을 위해 클래스 이름 값 할당
            pred_cls = [classes[x] for x in pred.cpu().squeeze().numpy()] #예측값 라벨
            print(target_cls, pred_cls, sep='\n')
            plot_tensor(data, mode=1, num_col=4, label=[target_cls, pred_cls]) #plot_tensor호출 매개변수 전달
            break
