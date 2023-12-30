import sys

def calcul_min(k):
    count = []
    cnt = 0
    for i in range(n):
        cnt = k // coins[i]
        count.append(cnt)
        k -= cnt*coins[i]
    print(sum(count))
        
n, k = map(int, sys.stdin.readline().split())

coins = []
for _ in range(n):
    coins.append(int(sys.stdin.readline()))

coins.sort(reverse=True)
calcul_min(k)
