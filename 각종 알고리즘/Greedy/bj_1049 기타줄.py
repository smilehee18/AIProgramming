import sys
'''
입력 : 구매하고자 하는 기타줄의 개수와 브랜드 개수
각 브랜드마다 6_패키지 기타줄과 낱개 기타줄의 비용
출력 : 기타줄 tar개를 구매할 때 최소 비용
[핵심]
6패키지/6 결과가 낱개로 구매하는 것보다 저렴하면
최대한 6패키지를 많이 구매하고, 나머지 기타줄은 나머지 연산 이용해서 비교
반대의 경우, 즉 6패키지/6 이 낱개로 구매하는 비용보다 비싸면
그냥 낱개비용 x 구매할 기타줄 수 만큼이 최소 비용이 된다
[알고리즘]
1 브랜드마다 (6패키지, 낱개) 기타줄의 비용을 튜플로 묶어서 리스트에 저장
2 for 문을 돌면서 6패키지, 낱개 비용의 최소비용을 갱신, 저장
3 min_6pa//6 과 min_1 값을 비교하여 cost (최소 비용) 갱신
'''
tar, n = map(int, sys.stdin.readline().split())

#li 초기화 및 입력받은거 튜플로써 할당
li = []
cost = 0
for i in range(n):
    li.append(tuple(map(int, sys.stdin.readline().split())))

min_6pa = li[0][0]
min_1 = li[0][1]
for i in range(n):
    if(min_6pa > li[i][0]):
        min_6pa = li[i][0]
    if(min_1 > li[i][1]):
        min_1 = li[i][1]

if(min_6pa//6 < min_1): #6개 패키지로 구매하는것이 더 저렴한 경우
    cost += (min_6pa*(tar//6)) #나누기 연산 -> 몫 x 6패키지 비용 
    #낱개로 구매 vs 6패키지 1개 구매 비교해서 작은값을 cost에 더함
    #즉, 나머지값 가지고 보다 저렴한 방식을 찾는다
    cost += min(min_6pa,(tar%6)*min_1) 
else: #낱개로 구매하는 것이 더 저렴한 경우 
    cost += (tar * min_1) #단순히 낱개 최소비용 x 기타줄 수 
print(cost)