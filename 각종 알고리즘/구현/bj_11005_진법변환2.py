import sys

#문자열 붙이고 싶을 때 += 연산자 가능
#join의 대상은 iterable한 리스트와 같은 객체(주의)
#문자 타입으로 바꾸고 싶으면 str()
#아스키 코드에 해당하는 문자로 바꾸고 싶으면 chr()

# A 아스키 65 -> A 10진수 10이므로 55 감소
def val_to_char(i):
    if 0 <= i <= 9:
        return str(i)
    else:
        return chr(i+55) 

#입력 시작
n, b = map(int, sys.stdin.readline().split())

i = 1
val = []
while(n>0):
    val.append(n%b)
    n = n // b

# print(val)

ans = ''
for i in reversed(val):
    ans += val_to_char(i)
    
print(ans)