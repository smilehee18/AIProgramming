import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

cost_history = []
accuracy_history = []

csv = pd.read_csv('heart_norm.csv') #csv 파일 불러오기
csv = torch.FloatTensor(csv.values) #텐서로 만들기
X = csv[:, :13] #텐서 분할하기 -> X값
Y = csv[:, 13:] #Y값(0 또는 1)

x_train, x_test, y_train, y_test = train_test_split(  #훈련, 예측할 자료 나누기
    X, Y, test_size=0.2, random_state = 2, shuffle=True
)

dataset = TensorDataset(x_train, y_train)                                     #dataset 객체 추가
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True) #dataloader 이용하여 16개씩 섞기

model = nn.Sequential(  #모델 input 13개 output 1개
    nn.Linear(13, 1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), momentum=0.5, lr=0.02)

num_epochs = 60
for epoch in range(num_epochs + 1):
    for idx, (xt, yt) in enumerate(dataloader):
         hypo = model(xt)                        #예측값 추정
         cost = F.binary_cross_entropy(hypo, yt) #정규화로 cost값 추정

         optimizer.zero_grad() #gradient 값 초기화 하기
         cost.backward()       #gradient decent 계산
         optimizer.step()      #w값, b값 갱신

         pred = (hypo >= torch.FloatTensor([0.5]))                  #확률 계산하기 0 또는 1의 값
         correct_pred = (pred.float() == yt)                        #예측값과 정답이 같다면
         accuracy = (correct_pred.sum().item() / len(correct_pred)) #accuracy구하기
         accuracy_history.append(accuracy)
         cost_history.append(cost.item())
         if(idx == 0):
             print('Epoch {:d} COST: {:.5f} Accuracy: {:.6f}%'.format(epoch, cost.item(), accuracy * 100))
         print(idx)
print('Learning finished!')

#그래프로 보여주는 코드
plt.rcParams['axes.grid'] = True
fig, axes = plt.subplots(2, 1)
axes[0].plot(cost_history, 'r--')
axes[0].set_title('cost')
axes[1].plot(accuracy_history, 'b--')
axes[1].set_title('accuracy')
plt.tight_layout()
plt.show()

print('Model Test Start!')
with torch.no_grad():
    y_pred = (model(x_test) >= torch.FloatTensor([0.5]))
    correct_pred2 = (y_pred.int() == y_test) #예측값과 정답값이 같은지 계산
    accur = correct_pred2.float().mean() #평균값계산
    print('Accuracy:', accur.item()) #정확도 출력하기
