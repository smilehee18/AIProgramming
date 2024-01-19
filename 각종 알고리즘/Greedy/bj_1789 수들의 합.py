import sys
'''
수들의 합 S가 입력으로 주어졌을 때, 서로 다른 수들의 최대 개수를 출력하는것
ex) 55 -> 1~10까지 10 출력
[핵심]
서로 다른 수가 핵심!!
서로 다른 수들의 최대 갯수를 세야 하므로,
단순히 1부터 2,3,4,5... 이런식으로 자연수들의 차이가 1씩인 수들을 더해준뒤(sum)
sum>s 값이면 break하고 i-1번째 를 출력(이미 커져 버린수이므로 i-1번째가 정답)
'''
s = int(sys.stdin.readline())

sum = 0
for i in range(1, s+1):
    sum += i
    if(sum > s):
        break
if(s==1):
    print(1)
else:
    print(i-1)