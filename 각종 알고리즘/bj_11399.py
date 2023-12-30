import sys

num = int(sys.stdin.readline())

p = list(map(int, sys.stdin.readline().split())) 
p.insert(0, 0)
p.sort() #오름차순으로 정렬

#돈을 인출하는데 필요한 시간의 최솟값 계산
min = 0
for i in range(1, num+1):
    p[i] = p[i] + p[i-1]
    min += p[i]

print(min)
