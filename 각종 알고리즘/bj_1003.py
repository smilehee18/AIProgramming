#피보나치 -> 순환으로 하면 시간초과 나서 DP로 해결해야함
'''
DP로 테이블에 적어놓으면 fibo 함수의 중복호출 횟수를 감소시킬 수 있다.
시간복잡도 : O(n)
'''
import sys

#테이블은 0부터 증가한다.
#밑바닥부터 증가
def fibo(k):
    #문제조건 : k는 0~40까지다
    li = [None for _ in range(41)]
    li[0] = 0 
    li[1] = 1
    for i in range(2, k+1):
        li[i] = li[i-1] + li[i-2]
    return li[k]

n = int(sys.stdin.readline())

for i in range(n):
    k = int(sys.stdin.readline())
    if(k == 0): 
        print(1, 0)
    elif(k == 1): 
        print(0, 1)
    else: 
        print(fibo(k-1), fibo(k))