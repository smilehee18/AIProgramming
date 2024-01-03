import sys
'''
핵심 알고리즘 : 리스트에서 가장 긴 증가하는 부분 수열을 찾기 위해서는 인덱스의 위치를 변화시키며 
나보다 작은 수가 몇개 인지를 찾아야 한다.
따라서 dp 리스트에 i번째 인덱스 값보다 작은 값의 개수+1(i 포함) 값을 적어두면 된다.
[놓친 부분]
dp 리스트에 어떤 값을 적어야 할지 생각하기
수열이 오름차순 정렬이 아니기 때문에
dp[i] 갱신시에 max(dp[i], dp[j]+1) 함으로써 i번째 인덱스 이전 값들중에서
가장 큰 값을 찾아야 한다.(계속 비교하는 효과)
'''
n = int(sys.stdin.readline())
seq = list(map(int, sys.stdin.readline().split()))

dp = [1 for _ in range(n)]
ans = 1
for i in range(n):
    val = seq[i] #수열에서 n번째 인덱스
    for j in range(0,i):
        cur = seq[j] #n보다 앞에 있는 수들을 모두 순회, 현재 인덱스
        if(val > cur): #내 앞에 수보다 내가 클 때마다
            dp[i] = max(dp[i], dp[j]+1) #dp[i] 갱신 -> 메모이제이션

print(max(dp)) #dp의 가장 큰 값(부분 수열의 길이 출력)
