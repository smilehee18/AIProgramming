'''
[놓친 부분] 
map(str, bag) -> bag 리스트를 묶어서 문자열로 변환해준다.
.join() 함수 익히기 -> ' '공백을 문자열 사이사이에 넣어줄 때 유용함 
'''
import sys

n, m = map(int, sys.stdin.readline().split())
bag = [i+1 for i in range(n)]

for i in range(m):
    i, j = map(int, sys.stdin.readline().split())
    bag[i-1], bag[j-1] = bag[j-1], bag[i-1]
    #print(bag)

print(' '.join(map(str, bag))) 
