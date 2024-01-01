import sys
'''
[알고리즘]
1. (시작시간, 끝시간) 튜플로 만들어서 List 저장
2. 끝시간 기준 오름차순 정렬
3. 리스트 0번째 요소 무조건 채택
4. 1번째 요소부터 순회하면서 검사하기 시작
*현재 채택된 회의 끝나는 시간보다 i번째 회의 시작 시간이 같거나 큰경우 채택
*카운트 증가(카운트는 1부터 시작)
[놓친 부분]
1. 0번째를 채택했으므로 1번째부터 순회해야됨
2. 회의들의 끝나는 시간이 모두 같을 경우에는 시작시간을 비교해서 오름차순해야함
*sort함수의 key 파라미터를 조작해주면 되었음
'''

n = int(sys.stdin.readline())

li = []
for _ in range(n):
    val = sys.stdin.readline().split() #val 이라는 리스트에 값들을 임시 저장
    li.append((int(val[0]), int(val[1]))) #튜플로써 리스트에 추가 

li.sort(key=lambda x:(x[1], x[0])) #끝나는 시간 기준으로 오름차순 정렬
#print(li)

cur = li[0]
cnt = 1
for i in range(1, n): #0번째는 무조건 채택이므로 루프 1부터 시작 
    if(cur[1] <= li[i][0]): #선택된 회의의 끝나는 시간이 i번째 회의의 시작 시간보다 작다면
        cur = li[i] #i번째 회의 채택
        cnt+=1 #카운트 증가 
        #print(cur)
print(cnt)