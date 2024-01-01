import sys
import heapq

n, m = map(int, sys.stdin.readline().split())
li = list(map(int, sys.stdin.readline().split()))

pq = [(idx, val) for idx, val in enumerate(li)]
heapq.heapify(pq)

for i in range(m):
    firmin = heapq.heappop(pq)
    secmin = heapq.heappop(pq)
    sum_val = firmin[1] + secmin[1]
    li[firmin[0]] = sum_val
    li[secmin[0]] = sum_val
    heapq.heappush(pq, (firmin[0], sum_val))
    heapq.heappush(pq, (secmin[0], sum_val))
    print(li)
print(sum(li))

'''
firmin = min(li)
fidx = li.index(firmin)
secmin = min(li[:fidx]+li[fidx+1:])
if(firmin == secmin): sidx = li.index(secmin)+1
else: sidx = li.index(secmin)
li[fidx] = firmin + secmin
li[sidx] = firmin + secmin
print(sum(li))
'''