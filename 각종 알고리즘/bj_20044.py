import sys

n = int(sys.stdin.readline())

w = list(map(int, sys.stdin.readline().split()))

w.sort() #1. 오름차순 정렬
res = float("INF")
for i in range(n): #2. 왼, 오 줄여나가면서 작은 값 출력
    res = min(res, w[i]+w[(2*n-1)-i])

print(res)