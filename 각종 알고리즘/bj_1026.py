import sys
'''
b의 요소들을 재배열하지 않고 sum(a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1])이 최소가 되게 하는 것이 목표
1. sum이 최소가 되려면 가장 큰 수와 가장 작은 수끼리 곱해야되겠지 -> a는 오름차순, b는 내림차순으로 정렬
2. min(sum)은 재배열하든, 안하든 결과가 동일하기 때문에 재배열 불가 조건은 무시해도 됨
* A를 B에 맞추어 재배열해서 출력한 결과나, 큰수*작은수 결과는 같기 때문에
'''
n = int(sys.stdin.readline())

a = list(map(int,sys.stdin.readline().split()))
b = list(map(int,sys.stdin.readline().split()))

a.sort()
b.sort(reverse=True)

sum = 0
for i in range(n):
    sum += a[i] * b[i]
print(sum)
