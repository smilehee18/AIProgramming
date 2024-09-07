import sys
# 정수 타입에 대한 아스키 코드를 알고 싶으면 ord()
# 주의 !! 진법 변환할때 일의 자리부터 변환해야되니까 reversed 반드시
# A 아스키 65 -> A 10진수 10이므로 55 감소
def char_to_value(ch):
    if '0' <= ch <= '9':
        return int(ch)
    else:
        return ord(ch) - 55
    
#문자열로 받은거
n, b = sys.stdin.readline().split()
b = int(b) #b진법 -> 정수
values = [char_to_value(char) for char in n]

i = 1
sum = 0
for val in reversed(values):
    sum += (i * val)
    i *= b
print(sum)