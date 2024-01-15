import sys

n, m = map(int, sys.stdin.readline().split())
bag = [0 for i in range(n)]

for _ in range(m):
    src, dst, val = map(int, sys.stdin.readline().split()) #1, 2, 3
    #src ~ dst범위에 val 값을 할당
    for i in range(src, dst+1):
        bag[i-1] = val
print(*bag) #bag 리스트의 모든 원소를 언패킹한다
'''
print(*bag)의 기능 : bag = [1, 2, 3, 4, 5] -> 1 2 3 4 5 출력
'''