import sys
'''
서브태스크 : 리터 당 가격이 최대 10,000, 거리가 최대 10,000
'''
num = int(sys.stdin.readline()) #도시 개수
km = list(map(int, sys.stdin.readline().split())) #도로 길이 
cost = list(map(int, sys.stdin.readline().split())) #리터당 가격

minCost = cost[0]
sum = 0

for i in range(0, num-1):
    if(minCost > cost[i]):
        minCost = cost[i]
    sum+=(minCost*km[i])
print(sum)

''' #subtask #1
cur_cost = cost[0] #0번째는 무조건 채택
cur_km = km[0]
sum = cur_cost * cur_km #최소비용에 0번째 비용*거리 할당
tmp_km = 0
i = 1
while(i < num-1): #도시가 n개이면 길은 n-1개
    tmp_km = 0
    cur_cost = cost[i] #i번째 비용
    cur_km = km[i] #i번째 거리 
    tmp_km += cur_km #i번째 거리 할당
    j = i+1 
    while(j < num-1 and cur_cost <= cost[j]): #현재 주유소의 비용보다 다음 주요소 비용이 더 크다면 손해
        tmp_km += km[j] #다음 주유소의 km까지 합해줌
        j += 1
    sum += (tmp_km * cur_cost)
    i = j #i를 j로 업데이트 
print(sum)
'''