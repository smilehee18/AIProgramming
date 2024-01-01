import sys
'''
핵심 알고리즘 : 문제의 점화식을 따져보면
n = 1 -> 1가지
n = 2 -> 2가지
n = 3 -> 3가지
n = 4 -> 5가지
n = 5 -> 8가지... 
즉, 피보나치 수열의 진행과정과 같다
*그런데, 
n = 1 -> 1가지
n = 2 -> 2가지 는 다르므로 따로 처리해줘야함 
'''
def calcul_case(n):
    #dp[0]의 경우 Index Error가 발생하는것을 방지
    dp = [None for _ in range(max(3, n+1))] #dp의 크기 최소 3(0, 1, 2)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 2 
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    print(dp[n]%10007)

n = int(sys.stdin.readline())
calcul_case(n)