import sys
INF = int(1e9)  # 무한대를 나타낼 값 설정

num_city = int(sys.stdin.readline())
num_line = int(sys.stdin.readline())

#그래프 생성
graph = [[INF for _ in range(num_city)] for _ in range(num_city)]
city_list = [[2, i] for i in range(1, num_city+1) for _ in range(num_city)]

#입력
for i in range(num_line):
    a, b, c = map(int, sys.stdin.readline().split())
    graph[a-1][b-1] = min(graph[a-1][b-1], c)

# 입력 후 바로 자기 자신으로 가는 경로는 0으로 설정
# A -> A, B -> B 등
for i in range(num_city):
    graph[i][i] = 0
    city_list[i*(num_city+1)] = [0]
    
#플로이드 알고리즘
for k in range(num_city):
    for i in range(num_city):
        for j in range(num_city):
            if(graph[i][j] > graph[i][k] + graph[k][j]):
                graph[i][j] = graph[i][k] + graph[k][j]
                city_list[(i*num_city)+j][0] += 1
                city_list[(i*num_city)+j].append(k+1)
            #graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
            
#출력
for i in range(num_city):
    for j in range(num_city):
        if(graph[i][j] == INF): print(0, end=' ')
        else: print("%d"%graph[i][j], end=' ')  
    print("")

print(city_list)

for i in range(len(city_list)):
    for j in range(len(city_list[i])):
        print("%d"%city_list[i][j], end=' ')
    if(len(city_list[i]) == 1): print("")
    else: print(((i+1)%num_city) if((i+1)%num_city>0) else num_city, end='\n')