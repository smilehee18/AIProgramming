import sys
'''
[놓친 부분]
인덱스 처리 잘하기 
fibo[0], fibo[1]일때 (기반상황)는 무족건 표에 포함되어 있어야 하므로
range 범위를 max(2, n+1)로 설정
'''
n = int(sys.stdin.readline())

fibo = [0 for _ in range(max(2, n+1))]
fibo[0] = 0
fibo[1] = 1
for i in range(2, n+1):
    fibo[i] = fibo[i-1] + fibo[i-2]
print(fibo[n])