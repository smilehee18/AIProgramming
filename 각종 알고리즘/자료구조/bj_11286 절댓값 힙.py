import sys
import heapq
'''
[놓친 부분]
우선순위를 주어서 힙큐를 이용하고자 할때,
heapq.heappush() 함수를 이용하여 튜플 형태로 data를 힙에 넣는다.
0번째 매개변수 : 해당 힙 리스트, 1번째 매개변수 : 튜플
튜플 첫번째 요소 : 우선순위를 결정하는 값 (여기서는 절댓값),
튜플 두번째 요소 : 실제 데이터
따라서 heapq.heappush((h), (abs(x), x))
'''
h = []
n = int(sys.stdin.readline())

for _ in range(n):
    x = int(sys.stdin.readline())
    if(x!=0): #x가 0이 아니면
        #튜플의 형태로 힙에 push한다 (우선순위, 실제 값)
        heapq.heappush(h, (abs(x), x))
    else: #x가 0이면 절댓값이 가장 작은 data pop
        if(len(h) == 0):
            print(0)
        else:
            val = heapq.heappop(h)[1] #튜플의 1번째 요소인 실제 값을 fetch
            print(val)