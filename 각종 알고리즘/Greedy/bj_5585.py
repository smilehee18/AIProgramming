import sys

def calcul_count(coin, coins):
    count = [] #잔돈 개수를 카운트 할 리스트
    for i in range(len(coins)):
        ccnt = coin//coins[i]
        count.append(ccnt)
        coin -= coins[i]*ccnt
    print(sum(count))

val = int(sys.stdin.readline()) #물건의 값
coins = [500, 100, 50, 10, 5, 1] #동전 종류
calcul_count(1000-val, coins)