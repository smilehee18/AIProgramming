import sys 

#리스트에 - 기준으로 나눈 수식 또는 피연산자가 저장됨
input_str = sys.stdin.readline().split('-')

#첫번째 수식만 놔두고 나머지것들은 싹다 빼는 원리
#tmp에 첫번째 수식이 저장됨
tmp = input_str[0].split('+')
ans = 0

for val in tmp:
    ans += int(val)

#0번째 수식은 제외하고!!
#input_str로 두는 이유는 - 구분자로 분리된 녀석들이 여러개일수 있음. 끼리끼리 더해줌
for i in range(1, len(input_str)):
   tmp = input_str[i].split('+') 
   #몽땅다를 + 기준으로 분리해주고 (tmp는 리스트임(원소 1개~n개))
   print(tmp)
   for val in tmp:
       ans -= int(val)

print(ans)
