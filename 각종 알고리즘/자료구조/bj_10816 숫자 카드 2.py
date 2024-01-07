import sys
from collections import Counter

def count_val(key, counter):
    return counter[key]

n = int(sys.stdin.readline())

nlist = list(map(int, sys.stdin.readline().split()))
m = int(sys.stdin.readline())
mlist = list(map(int, sys.stdin.readline().split()))

nlist.sort() #오름차순 정렬 
counter = Counter(nlist)
#print(counter) #Counter({10: 3, -10: 2, 3: 2, 2: 1, 6: 1, 7: 1})
cnt = []
for i in range(m):
    val = mlist[i]
    idx = count_val(val, counter)
    cnt.append(idx)
print(' '.join(str(i) for i in cnt))