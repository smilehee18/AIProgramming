import sys
'''
스네이크의 처음 길이 l이 주어졌을 때, 
스네이크가 과일을 하나씩 먹고 나서 늘릴 수 있는 최대 길이를 구하기
핵심은 [늘릴 수 있는 최대 길이]
먹이 리스트를 오름차순해서 작은 순서대로 나열하면->
먹을 수 있는 먹이의 개수가 많아짐 -> 스네이크 길이를 늘릴 수 있음
'''
n, l = map(int, sys.stdin.readline().split())
li = list(map(int, sys.stdin.readline().split())) #먹이 리스트

li.sort() #먹이 리스트 오름차순 정렬

for i in range(n):
    if(l >= li[i]): #i번째 먹이가 내 길이보다 작거나 같으면
        l+=1 #스네이크 길이 1 증가 

print(l)