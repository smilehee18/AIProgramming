import sys
'''
how? DP 테이블을 만들어서 중복 연산 방지, 메모리 효율적 사용
n <= 10 조건이 주어졌으므로 11x11 table 생성
기반 상황 및 일반상황에 따라 분기문 작성
'''
n, k = map(int, sys.stdin.readline().split())

dp = [[-1 for _ in range(11)] for _ in range(11)]

for i in range(n+1):
    for j in range(min(i,k)+1): #nCr에서 r이 n보다 커지는 것 방지
        if(i == j or j == 0): #기반상황 
            dp[i][j] = 1 
        else: #일반상황
            dp[i][j] = dp[i-1][j-1] + dp[i-1][j]

print(dp[n][k])