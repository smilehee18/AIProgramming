import sys

def calcul_resident(floor, ho):
    #2D 배열 아파트 크기만큼 생성 
    apt = [[j for j in range(ho+1)] for i in range(floor+1)]

    apt[1][0] = 0 #1층 0호 0 초기화 
    for i in range(1, floor+1):
        for j in range(1, ho+1):
            #i층 j호는 다음 규칙을 따른다.
            apt[i][j] = apt[i][j-1] + apt[i-1][j]
    print(apt[floor][ho])
n = int(sys.stdin.readline())

for i in range(n):
    floor = int(sys.stdin.readline()) #층
    ho = int(sys.stdin.readline()) #호
    calcul_resident(floor, ho)