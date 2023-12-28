import sys

#동적계획법 : 이전에 적어놓았던 것을 활용하여 테이블을 써내려간다.
#중요한 점화식 : (3원동전을 써서 6원을 만들거야) 
#-> DP[i-1(이전 동전)][6원동전-0]+DP[i-1(이전 동전)][6원동전-3*1]+DP[i-1(이전 동전)][6원동전-3*2] + ..
'''
핵심 알고리즘
1. x축은 target(만들어야 하는 금액), y축은 동전의 종류로 DP TABLE을 만든다.
2. (0, 0)부터 시작하여 루프를 돌면서 이전에 적어놓았던 경우의 수를 기반으로 테이블을 채워나간다.
3. 동전의 개수가 몇 개 필요하니? 를 구하는 cnt 변수와 dp[i-1][j-k]를 생각하는게 중요함
*dp[i-1][j-k]에서 k는 가장 안쪽 루프에서 0~coin*cnt+1까지 coin만큼 증가한 값으로 3원이라면 3의 배수임*
'''
def knapSack_dp(target, val, num):
    dp = [[0] * (target+1) for _ in range(num+1)] #2D DP 테이블을 생성
    dp[0][0] = 1 #0원을 만드는 방법은 1가지가 있다.
    #테이블을 채워가면서 target원을 만드는 거야
    #그러니까 테이블의 위치는 j(column)원을 만들 수 있는 경우의 수임
    #경우의 수를 고려하면서 테이블을 채워간다.
    for i in range(1, num+1):
        coin = val[i-1] 
        for j in range(0, target+1):
            cnt = j//coin #현재 j원에서 가질 수 있는 동전 몇 개? cnt 계산
            for k in range(0, coin*cnt+1, coin):
                dp[i][j] += dp[i-1][j-k]
    
    print(dp[num][target])


n = int(sys.stdin.readline()) #테스트 케이스 개수

for _ in range(n):
    num = int(sys.stdin.readline()) #동전의 가지수
    val = list(map(int, sys.stdin.readline().split())) #동전의 각 금액(배낭에서의 무게들 리스트)
    target = int(sys.stdin.readline()) #만들어야 하는 금액 
    knapSack_dp(target, val, num)