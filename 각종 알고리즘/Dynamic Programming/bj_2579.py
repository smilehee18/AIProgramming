import sys

n = int(sys.stdin.readline()) #계단 수
val = []
for _ in range(n):
    val.append(int(sys.stdin.readline())) #계단 가치 리스트

dp = [0 for _ in range(n)] #테이블

#인덱스 처리 중요
dp[0] = val[0] #1칸 올라간 경우
if(n > 1):
    dp[1] = max(val[1], val[0]+val[1]) #2칸 올라간 경우 
if(n > 2):
    dp[2] = max(val[0]+val[2], val[1]+val[2]) #3칸 올라간 경우

for i in range(3, n):
    #4칸부터(인덱스-1)는 점화식을 따른다
    dp[i] = max(dp[i-2]+val[i], dp[i-3]+val[i-1]+val[i])
print(dp[n-1])