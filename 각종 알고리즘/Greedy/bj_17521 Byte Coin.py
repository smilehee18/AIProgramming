import sys
'''
[핵심 알고리즘]
시간에 따른 코인 주가 그래프를 보고 살지 말지를 판단
1) i+1 > i 이면 코인을 사는게 이득이므로 w//coins[i] 개수만큼 산다
-> 왜 ? : 주가가 올라가기 전에 사야 더 많은 코인을 살 수 있으니까
2) i+1 < i 이면 코인을 파는게 이득이므로 코인 개수만큼 팔아서 현금 이득 얻음
-> 왜 ? : 주가가 내려가기 전에 팔아야 더 많은 현금을 취할 수 있으니까 
*for 문은 (n-1)범위만큼 동작하므로, 마지막에 코인들을 팔지 않았을 때를 대비하여
isBuy가 True일 때, 가지고 있는 코인을 다 팔아버림
[놓친 부분]
코인 주가를 보고 i < i+1일때 코인을 사야 한다는 것을 파악하고
주가 상승 시점을 기준으로 리스트를 분리해서 접근하려고 했었다.
문제점은 코인을 팔아야 하는 시점을 정확하게 짚지 못하여서 헤멨다
바뀌는 시점(사는 시점, 파는 시점)을 정확히 파악하자
'''
coin = []
series = []
n, w = map(int, sys.stdin.readline().split())

for _ in range(n):
    coin.append(int(sys.stdin.readline()))

start = 0
isBuy = False
for i in range(n-1):
    #코인 구매해야 할 때 
    if(coin[i] < coin[i+1] and not isBuy):
        coins = w // coin[i]    #살 수 있는 코인의 양  
        w = w - (coin[i]*coins) #남은 돈 액수
        isBuy = True
    #코인 팔아야 할 때
    elif(coin[i] > coin[i+1] and isBuy):
        w = w + (coin[i] * coins) #현금으로 보는 이득 
        isBuy = False
    #print(w)
#코인을 다 사고 마지막에 안 팔았을 경우
if(isBuy == True):
    w = w + (coin[-1]*coins)
print(w)