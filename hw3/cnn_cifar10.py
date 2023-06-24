import torch
import torch.nn as nn
import torchvision                           #영상 관련 lib import
import torchvision.transforms as transforms  #데이터 증강을 위한 lib
import torch.optim as optim
import os                                    #폴더 구성을 위한 lib

#얼마나 정확하게 훈련되었는지 측정하는 함수
def accuracy(output, target, topk=(1, )): #tensor 메소드: 위~1등까지 순위를 매긴다
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        num_cor = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            num_cor.append(correct_k.clone())
            acc.append(correct_k.mul(1/batch_size))
    return acc, num_cor

#가중치를 초기화하는 함수
def init_weights(m, init_type='normal', init_gain=0.02):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal': #normal분포로 weight값을 초기화할게
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier': #xavier 성질로 weight값을 초기화할게
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal': #직교성질을 이용해서 weight값을 초기화할게
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0) #bias값을 0으로 초기화할게
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

#CNN Custom Layer 클래스 정의 및 구현부
class Network(nn.Module):
    def __init__(self): #생성자의 역할
        super(Network, self).__init__() #부모 메소드 상속 받고
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), (1,1)) #입력 채널 32, 커널 갯수 64, 커널 사이즈 3x3
        self.conv3 = nn.Conv2d(64, 128, 3, 1) #입력 채널 64, 커널 갯수 128, 커널 사이즈 3x3
        self.fc_code = nn.Linear(in_features=128, out_features=128) #입력 특징 128, 출력도 128
        self.fc_output = nn.Linear(128, 10) #입력 128, 출력 10 -> 총 10개의 클래스로 분류
        self.relu = nn.ReLU(inplace=True) #reLu activation 메소드를 이용하여 값자체를 바꿀게
        self.maxpool = nn.MaxPool2d(2)    #maxPool 방식을 이용하여 feature map의 크기를 줄임
        self.dropout = nn.Dropout(p=0.5)  #dropout응 이용하여 overfitting 방지
        self.apply(init_weights)          #가중치 초기화 -> normal 가우시안 분포 구성

    def forward(self, x):                                   # layer의 순서 부분
        feature_map = self.conv1(x)                         #convolution 연산을 한다
        activated = self.relu(feature_map)                   #활성화 함수를 씌운다
        compressed = self.maxpool(activated)                #pooling을 통해 feature map의 크기를 감소
        x = self.maxpool(self.relu(self.conv2(compressed))) #한번에 연산한것
        x = self.maxpool(self.relu(self.conv3(x)))          #conv3을 통해 한번에 연산한 것
        x = x.view(x.size(0), -1)                           #view로 사이즈를 바꾸어서 일자로 쭉 핀다
        code = self.dropout(self.relu(self.fc_code(x)))     #dropout 계산하기
        output= self.fc_output(code)                        #128개 -> 10개 확률
        return output, code                                 #확률과 128개의 특징값 추출

if __name__ == '__main__': #메인함수
    if not os.path.isdir("./ckpt_0"):  #폴더가 없으면
        os.makedirs("./ckpt_0")        #해당 폴더를 만들어 주세요.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #디바이스 정하기
    print(device)

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    transform_test = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    model = Network() #customNetwork 초기화 및 선언
    model.to(device) #연산장치 위에 올리기

    params = model.parameters() #나의 모델의 파라미터 받기

    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.001) #Adam 활성화 함수
    train_criterion = nn.CrossEntropyLoss(reduction='mean') #평균값으로 계산할게요.

    for epoch in range(1, 100): #1~99까지 반복
        model.train() #dropout 때문에 반드시 명시해주어야 함
        for step, (data, targets) in enumerate(trainloader): #trainloader의 값을 반복하며 돈다.
            data = data.to(device, dtype=torch.float) #계산할때는 cpu나 gpu위에 올린다
            targets = targets.to(device) #target도 역시 올린다
            optimizer.zero_grad() #optimizer 초기화

            outputs, code = model(data)  #특징값과 확률을 뽑아서
            loss = nn.CrossEntropyLoss(reduction='mean')(outputs, targets)
            #손실값을 구한다.(output은 모델로 예측한 엉뚱한 값, targets은 원래 정답값)
            loss.backward() #gradient 계산
            optimizer.step() #w값 갱신

            loss = loss.item() #손실값 할당
            acc, _ = accuracy(outputs, targets) #ouput과 정답값 사이의 정확도 계산
            acc = acc[0].item() #정확도 할당

            if step % 10 == 0: #10번 돌 때마다
                print('Epoch {} Step {}/{} Loss {:.4f} Accuracy {:.4f}'.format(epoch, step, len(trainloader), loss, acc))

        model.eval()    #여기서부터 test함
        total_cor = 0
        total_samples = 0

        with torch.no_grad(): #gradient 적용X
            for step, (data, targets) in enumerate(testloader):
                data = data.to(device, dtype=torch.float)
                targets = targets.to(device)
                outputs, code = model(data)               #확률값과 feature값
                _, num_cor = accuracy(outputs, targets)   #정확도 측정
                num_cor = num_cor[0].item()
                total_samples += data.size(0)
                total_cor += num_cor
            acc = total_cor / total_samples
            print('Epoch {} : Accuracy {:.4f}'.format(epoch, acc))  #에폭당 정확도 출력
        path = 'ckpt_0/model_state_%d.st'%(epoch) #모델의 가중치 w값 저장
        torch.save(model.state_dict(), path)      #해당 경로로 저장