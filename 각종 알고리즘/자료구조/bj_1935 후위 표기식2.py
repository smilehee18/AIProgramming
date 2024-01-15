import sys
#from collections import deque
'''
[알고리즘]
후위 표기식을 스택을 이용해 계산해서 출력하는 것이 목표
피연산자 -> 스택에 PUSH / 연산자 -> 스택 상단에서 2개 꺼내서 계산후 다시 PUSH
주의점) 스택 가장 최상단은 2번째 피연산자, 그 다음에 뽑아낸게 첫번째 피연산자가 됨
즉 스택에서 pop하면 sec, fir 순서 -> fir + sec 계산
deque 모듈로 스택을 구현한 결과 시간과 메모리 사용량이 더 많아졌다..
'''

numbers = []
stack = []
#stack = deque()
n = int(sys.stdin.readline())
str = sys.stdin.readline().strip() #문자열 입력
for _ in range(n):
    numbers.append(float(sys.stdin.readline()))

for char in str:
    #char가 연산자인 경우
    if(char=='*' or char=='+' or char=='-' or char=='/'):
        sec = float(stack.pop())
        fir = float(stack.pop())
        if(char=='*'): stack.append(float(fir*sec))
        elif(char=='+'): stack.append(float(fir+sec))
        elif(char=='-'): stack.append(float(fir-sec))
        else: stack.append(float(fir/sec))
    else: #char가 피연산자인 경우 
        stack.append(numbers[ord(char)-65])

print("%.2f" %(stack[0]))