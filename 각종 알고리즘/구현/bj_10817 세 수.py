import sys

li = list(map(int, sys.stdin.readline().split()))

li.sort()
print(li[1])