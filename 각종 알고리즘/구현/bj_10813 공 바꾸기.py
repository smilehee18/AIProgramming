import sys

n, m = map(int, sys.stdin.readline().split())
bag = [i+1 for i in range(n)]

for i in range(m):
    i, j = map(int, sys.stdin.readline().split())
    bag[i-1], bag[j-1] = bag[j-1], bag[i-1]
    #print(bag)

print(' '.join(map(str, bag)))