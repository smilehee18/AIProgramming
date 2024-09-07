#플로이드 알고리즘의 경로 출력
#city_list는 경로를 담고 있는 2차원 리스트
#for문의 if 문에서 graph 뿐만 아니라 경로 리스트도 함께 갱신
#city_list[i][k][:-1] : 마지막 요소 제외 : 나는 빠질게
#print(len(city_list[i][j]), *city_list[i][j])
#*연산자 : 언패킹(리스트의 요소들을 공백으로 분리하여 출력가능) 
#실수 : 경로의 개수까지 합쳐서 표현 X -> len을 이용하면된다.

import sys
INF = int(1e9)  # 무한대를 나타낼 값 설정

num_city = int(sys.stdin.readline())
num_line = int(sys.stdin.readline())

# 그래프 생성
graph = [[INF for _ in range(num_city)] for _ in range(num_city)]
city_list = [[[] for _ in range(num_city)] for _ in range(num_city)]

# 입력
for i in range(num_line):
    a, b, c = map(int, sys.stdin.readline().split())
    graph[a-1][b-1] = min(graph[a-1][b-1], c)
    city_list[a-1][b-1] = [a, b] #(1, 2) (1, 5) 저장

# 자기 자신으로 가는 경로는 0으로 설정
for i in range(num_city):
    graph[i][i] = 0
    city_list[i][i] = [i+1]

# 플로이드 워셜 알고리즘
for k in range(num_city):
    for i in range(num_city):
        for j in range(num_city):
            if graph[i][j] > graph[i][k] + graph[k][j]:
                graph[i][j] = graph[i][k] + graph[k][j]
                # 경로 갱신
                #city_list[i][k][:-1] -> k인 자기 자신 빼고 리스트의 요소를 합치기 위해
                city_list[i][j] = city_list[i][k][:-1] + city_list[k][j]

# 출력
for i in range(num_city):
    for j in range(num_city):
        if graph[i][j] == INF:
            print(0, end=' ')
        else:
            print(graph[i][j], end=' ')
    print()

# 경로 출력
for i in range(num_city):
    for j in range(num_city):
        # 자기 자신이라면 0 출력
        if graph[i][j] == INF or i == j:
            print(0)
        else:
            #*연산자 : unpacking 리스트 안의 요소들을 공백 기준으로 풀어주는 역할
            #1. 경로 개수 출력, 2. 경로 출력
            print(len(city_list[i][j]), *city_list[i][j]) 
