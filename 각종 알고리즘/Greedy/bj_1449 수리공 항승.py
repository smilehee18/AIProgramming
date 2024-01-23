import sys

n, l = map(int, sys.stdin.readline().split())
water = list(map(int, sys.stdin.readline().split()))

water.sort() #물이 새는 위치 리스트 오름차순 정렬

cnt = 1 #0번째 위치는 무조건 테이프 붙여야 함 
start = water[0] #0번째위치 = 시작점 
# x+0.5+0.5 <= l 만족해야 테이프 1개 붙일 수 있다
# x + 1 <=l, x <=1-1 조건 
for i in range(1, n): #시작점이 0이므로 1-0, 2-0, 3-0 순으로 비교
    if(water[i]-start > l- 1): #만약 1개 테이프로 붙일 수 없는 거리이면
        cnt+=1 #테이프 개수 증가(겹쳐서 붙임)
        start = water[i] #시작점 0 -> 1로 변경 (붙였으므로)

print(cnt) #테잎의 최소 개수 출력 