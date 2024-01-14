import sys
'''
조건 : 입력은 알파벳 소문자로만 이루어져 있다.
문제 : 알파벳 개수 카운트해서 출력하기 
놓친부분 : alpha[i-ord('a')]의 인덱스 변환 부분 파악. ch - '0'을 떠올리자!
ord() 함수 : 인자로 받은 문자를 아스키 코드 (정수값)으로 반환해줌
count() 함수 : str.count(i)는 i가 몇 번 카운팅 되었는지를 반환해줌
'''
str = sys.stdin.readline()
alpha = [0] * 26

#a~z까지 26번 반복, ord()함수는 해당 문자를 아스키 코드 값으로 반환
for i in range(ord('a'), ord('z')+1):
    alpha[i - ord('a')] = str.count(chr(i)) #count 함수이용, 문자 세기 
    '''
    예를 들어, 입력 문자열이 "abc"라면 alpha[97-97], alpha[0] = 1
    alpha[98-97] = 1, alpha[1] = 1, 
    alpha[99-97] = 2, alpha[2] = 1
    '''
#공백 넣어서 alpha 리스트 프린트
for i in alpha:
    print(i, end=' ')
