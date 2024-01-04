'''
Queue 모듈 -> 시간복잡도 O(N)
deque 모듈 -> 양쪽 끝에서 아이템을 추가하거나 제거 가능 O(1)
'''
from collections import deque
import sys

n = int(sys.stdin.readline())
deq = deque(range(1, n+1)) #deq initialize

while len(deq) > 1:
    deq.popleft()  # 가장 상단 카드 제거
    deq.rotate(-1)  # 덱을 왼쪽으로 한단계 회전 

print(deq[0])
'''
from queue import Queue
import sys

n = int(sys.stdin.readline())

que = Queue()
for i in range(n): #큐 initialize
    que.put(i+1)

while(True):
    que.get() #가장 상단 카드 제거 
    if(que.qsize() == 1):
        break
    val = que.get()
    que.put(val)

print(que.get())
'''