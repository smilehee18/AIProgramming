import sys
'''
[알고리즘]
* a->b 만들기 위해서는 x*2를하거나 숫자 끝에 1을 더하거나 이므로
  b->a로 갈때 b%2==0, b%10==1 여부를 검사하면 된다 *
  (a < b 범위 내에서) 
[놓친 부분]
b->a로 가는 while문 조건을 (a<b)로 설정했으므로 
b%2 ==0, b%10==1 둘 다 만족하지 않는 경우의 분기문도 필요하다.
-> break, while문 빠져나와서 a==b 여부 검사하는 것으로 수정
(반례) 1, 131 -> -1 출력 
'''
a, b = map(int, sys.stdin.readline().split())

cnt = 0
while(a < b): 
    if(b%10 == 1): #끝자리가 1이면
        b //= 10   #끝자리 수 제거
        cnt+=1  
    elif(b%2==0):
        b //= 2
        cnt+=1
    else:
        break

if(a == b):
    print(cnt+1)
else:
    print(-1)