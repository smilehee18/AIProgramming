import sys
from collections import Counter

str = list(sys.stdin.readline().strip())

str = [val.upper() for val in str]
commons = [0]*26 #빈도수 알파벳 리스트(A~Z) 
#빈도수를 리스트에 저장해놓아서 리스트의 가장 큰 요소를 출력하는 풀이법
#ord 함수는 알파벳을 해당 인덱스로 변환해준다. ch - '0' 이랑 비슷한 원리!!
for s in str:
    commons[ord(s)-ord('A')] += 1

max_val = max(commons)
if(commons.count(max_val) > 1):
    print('?')
else:
    print(chr(commons.index(max_val)+ord('A'))) #A만큼의 수를 더해서 알파벳 숫자로 나타내준 뒤, char형으로 바꿈
    
#collection 모듈을 이용한 풀이법 
'''[알고리즘]
가장 빈도수가 높은 수가 2개 이상이면 ? 문자를 출력해야 하므로
collection의 counter 모듈을 이용해 가장 빈도수가 높은 수 2개를 뽑아옴
그러면, common 리스트 안에 튜플로써 [('문자', 빈도수), ('문자', 빈도수)] 이렇게 저장됨 
만약 0번째 빈도수와 1번째 빈도수가 같다면 ? 출력, 
다르다면 1순위가 정해져 있다는 것이므로, 0번째 문자를 출력하는 원리
[놓친 부분]
입력에서 문자 하나만 주어지는 경우 고려해야함 
입력 문자 리스트 요소가 2개인데 둘 다 같은 빈도수 (1, 1)인 경우
most_common(2)가 반환하는 요소는 결국 1개이기 때문에
len(common) < 2 인 경우 고려
다양한 케이스를 생각해보는 연습을 하자..
'''
''' 
if(len(str) == 1):
    print(str[0])
else:
    counter = Counter(str)
    common = counter.most_common(2)

    if(len(common) < 2 or common[0][1] != common[1][1]):
        print(common[0][0])
    else:
        print('?')
'''