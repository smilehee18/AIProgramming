import sys
'''
[핵심 알고리즘]
가장 마지막 레벨이 가장 높은 점수여야 하므로,
가장 마지막 레벨 점수 부터 비교하기 시작 (거꾸로 비교)
for문 돌아갈 때마다 직전 값 i-1 이 현재 값 i보다 크거나 같다면
ans에 last - first + 1 만큼을 ans에 더해주고 (간격, 차이)
i-1 번째 수를 갱신한다
[내가 생각했던 거]
마지막 수가 가장 큰 수이므로,
마지막 수 기준으로 처음 수부터 n개만큼 1씩 증가하는 등차수열 리스트를 만들고
원래 리스트와 내가 만든 리스트와의 차를 ans에 더해줬다
하지만 이 방법의 문제점은 최소 점수down 조건을 만족 못함
이유 : 입력 들어온 수가 어떤 수인지 모르므로, 무조건 1씩 증가하는 등차수열
을 만족하게 하는 것은 논리적 X. 입력에 맞춰야대
'''

li = []
n = int(sys.stdin.readline())

for _ in range(n):
    li.append(int(sys.stdin.readline()))

ans = 0
#n-1부터 0까지 -1씩 ++
for i in range(n-1, 0, -1):
    if(li[i] <= li[i-1]):
        diff = li[i-1] - li[i] + 1
        li[i-1] = li[i] - 1
        ans += diff
print(ans)
