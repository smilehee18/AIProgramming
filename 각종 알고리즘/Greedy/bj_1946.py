import sys
'''
핵심 : 다른 모든 지원자와 비교했을 때, 적어도 한 명의 지원자보다
(면접점수 and 서류 점수)가 낮다면 탈락
--> 선발가능한 신입사원의 최대 인원수를 구하는 것이 목적..
How? : 서류 등수 기준으로 오름차순으로 정렬했을 때, 
0번째 지원자는 이미 합격(서류 1등이므로).
minspk 변수를 두어서 0번째 지원자의 면접 등수를 할당,
n 크기만큼 순회하면서 minspk(최소 면접 등수)보다 낮은 등수를 가진 지원자가 있다면
해당 지원자는 합격이므로 카운트 증가 & minspk 재할당
-> 이렇게 하는 이유는 서류 등수 오름차순으로 정렬했으므로
나보다 앞의 지원자들은 이미 서류 등수가 높으므로 볼 필요가 X,
최소 면접 등수 기준으로 나머지 지원자들의 면접 등수 검사 및 갱신해주면 된다. 
'''
test = int(sys.stdin.readline())

def count_Num(num, li):
    survive = 1 #0번째는 이미 뽑혔으므로(서류등수 1위)
    minspk = li[0][1] #서류 점수 1위 신입 기준 최소 면접 등수 선언
    for i in range(1, num):
        if(li[i][1] < minspk): #현재 면접 등수가 더 낮다면(점수 더 높은거)
            survive +=1        #해당 신입사원 채택  
            minspk = li[i][1]  #최소 면접 등수 갱신
    print(survive)

for _ in range(test):
    li = []
    num = int(sys.stdin.readline())
    for _ in range(num):
        a, b = map(int, sys.stdin.readline().split())
        li.append((a, b))
    li.sort(key=lambda x:x[0]) #서류 등수 낮은 친구부터 오름차순
    count_Num(num, li)