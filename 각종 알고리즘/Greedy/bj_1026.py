import sys

n = int(sys.stdin.readline())

a = list(map(int,sys.stdin.readline().split()))
b = list(map(int,sys.stdin.readline().split()))

a.sort()
b.sort(reverse=True)

sum = 0
for i in range(n):
    sum += a[i] * b[i]
print(sum)