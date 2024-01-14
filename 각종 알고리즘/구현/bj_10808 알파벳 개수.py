import sys
'''
조건 : 알파벳 소문자로만 이루어져 있다.
알파벳 개수 세기
'''
str = sys.stdin.readline()
alpha = [0] * 26

#a~z까지 26번 반복, ord()함수는 해당 문자에 대한 인덱스 반환(like 아스키)
for i in range(ord('a'), ord('z')+1):
    alpha[i - ord('a')] = str.count(chr(i)) #count 함수이용, 문자 세기 

#공백 넣어서 alpha 리스트 프린트
for i in alpha:
    print(i, end=' ')