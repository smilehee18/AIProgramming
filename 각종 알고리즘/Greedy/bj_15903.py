import sys
import heapq
'''
[카드 합체 알고리즘]
x번, y번 카드 두 장에 쓰여진 값을 더한다.
계산 값을 두 장의 카드에 덮어쓴다.
카드 합체 이후 카드 값들의 합이 최소가 되게 하는 것이 목적이므로 
리스트에 카드값들을 저장 해놓고 꺼낼 때 가장 작은 수, 그다음으로 작은 수를 꺼내어 합하기
우선순위 큐를 이용해서 가장 작은 수 2개를 꺼냈다가 처리하고 다시 push하였다.
이때, 큐에서 꺼냈다가 다시 동일한 인덱스에 삽입해야 되므로 (숫자, 인덱스) 형식으로 heapq 요소 구성
[놓친 부분]
1. 처음에는 min()함수와 index()함수를 이용하고자 하였는데
   이렇게 되면 모든 요소들을 모두 순회하기 때문에 시간 복잡도 증가 -> 우선순위 큐 O(logN)
2. enumerate 모듈은 list에 있는 값들을 (idx, val) 순으로 밷어준다. -> 힙큐에 (val, idx) 삽입
-> 여기서 (idx,val)로 하게 되면 인덱스를 기준 작은 튜플을 pop하기 때문에 (val, idx)로 해야 제대로된 결과 도출
'''
n, m = map(int, sys.stdin.readline().split())
li = list(map(int, sys.stdin.readline().split())) #[3, 2, 6] -> 리스트로 

pq = [(val, idx) for idx, val in enumerate(li)] #(숫자 , 인덱스)
heapq.heapify(pq) #heapq (우선순위 큐) 정의

for i in range(m):
    firmin = heapq.heappop(pq) #가장 작은 val값 튜플 꺼냄
    secmin = heapq.heappop(pq) #2번째로 작은 val값 튜플 꺼냄
    #print(firmin, secmin)
    sum_val = firmin[0] + secmin[0] #0번째 인덱스 == val값 -> 더한다.
    li[firmin[1]] = sum_val         #li 리스트를 갱신시켜줌 (1번째 인덱스 == index값) 
    li[secmin[1]] = sum_val 
    heapq.heappush(pq, (sum_val, firmin[1])) #우선순위 큐에 다시 값을 넣음 (val, index)
    heapq.heappush(pq, (sum_val, secmin[1])) #리스트 개수는 변하지 x -> 값 다시 push
    #print(li)
print(sum(li)) #카트 합체 과정 이후 li 리스트 값들의 합 == 점수 

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