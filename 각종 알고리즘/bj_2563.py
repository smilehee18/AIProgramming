import sys

paper = [[0 for _ in range(101)] for _ in range(101)]
n = int(sys.stdin.readline())

for _ in range(n):
    x, y = map(int, sys.stdin.readline().split()) #x, y 좌표 입력받 
    for i in range(x, x+10): #색종이는 10 by 10이므로
        for j in range(y, y+10):
            paper[i][j] = 1

ans = 0
for rows in paper:
    ans += rows.count(1) #1인 놈들을 count한다.
print(ans)