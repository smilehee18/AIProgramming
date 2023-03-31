import cv2
import torch
import numpy as np

class TensorManipulator:  #클래스이므로 생성자 필요, 멤버 메서드 필요
    #TODO: 작성 필요
    #1) numpy array -> tensor 로 변환
    def __init__(self, img1, img2, img3): #생성자에서 초기화
        self.t_img1 = torch.tensor(img1)
        self.t_img2 = torch.tensor(img2)
        self.t_img3 = torch.tensor(img3)
        #print(self.t_img1.shape)
    #2) (480, 640, 3) 텐서 -> (3, 3, 480, 680) 크기의 텐서를 반환
    def concatenation(self):
        #각 이미지의 차원을 0을 기준으로 증가
        self.t_img1 = torch.unsqueeze(self.t_img1, dim = 0)
        self.t_img2 = torch.unsqueeze(self.t_img2, dim = 0)
        self.t_img3 = torch.unsqueeze(self.t_img3, dim = 0)
        #cat 이미지 결합
        tcat = torch.cat([self.t_img1, self.t_img2, self.t_img3], dim=0)
        #permute 차원 재배치
        tcat = tcat.permute(0,3,1,2)
        return tcat
    #3) (3, 3, 480, 640) 텐서 -> (3, 921600) 크기로 변환하여 반환
    def flatten(self, tcat):
        tshape = tcat.reshape([3, 921600])
        return tshape
    #4) 각 이미지 텐서들의 평균을 반환
    def average(self, tshape):
        tshape = tshape.type(dtype=torch.FloatTensor)
        return tshape.mean(dim = 1)
    pass

if __name__ == "__main__":
    img1 = cv2.imread('./1.jpg', cv2.COLOR_BGR2RGB)  # ndarray: (480, 640, 3)
    img2 = cv2.imread('./2.jpg', cv2.COLOR_BGR2RGB)  # ndarray: (480, 640, 3)
    img3 = cv2.imread('./3.jpg', cv2.COLOR_BGR2RGB)  # ndarray: (480, 640, 3)

    if (img1, img2, img3) is not None:
        cv2.imshow('1.jpg', img1)
        cv2.imshow('2.jpg', img2)
        cv2.imshow('3.jpg', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()

    obj = TensorManipulator(img1, img2, img3)
    out = obj.concatenation()
    out_flt = obj.flatten(out)
    out_avg = obj.average(out_flt)

    print(out.shape)
    print(out_flt.shape)
    print(out_avg)

