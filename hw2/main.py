import torch
import pandas as pd                     #csv 파일을 읽어오기 위해 import
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WineDataset(Dataset):
    #생성자에서 텐서의 모양 초기화
    def __init__(self, csv_path):
        self.csv = pd.read_csv(csv_path)
        self.csv = torch.Tensor(self.csv.values) #csv파일의 값만 뽑아와서 텐서형으로 할당한다.
        self.t1, self.t2 = torch.split(self.csv, 11, dim=1) #dim=1을 기준으로 1~11열 data와 12열 data를 분리
        self.t2 = self.t2.squeeze() #2차원 -> 1차원으로 차원 줄이기
        #print(self.t1.shape) #1599x11
        #print(self.t2.shape) #1599

    #객체 인덱스 접근 위함
    def __getitem__(self, idx):
        return self.t1[idx], self.t2[idx]

    #객체의 크기 반환
    def __len__(self):
        return len(self.t1) #1599

dataset = WineDataset('winequality-red-rev.csv')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #배치 크기 8, 한번에 8개 데이터를 섞어서 가져온다
x_batch, y_batch = next(iter(dataloader)) #8개 data를 각각 x_batch y_batch 텐서에 쏙쏙 할당
print(dataloader)
print(x_batch.shape) #8,11
print(y_batch.shape) #8
#print(x_batch); print(y_batch) #data를 출력