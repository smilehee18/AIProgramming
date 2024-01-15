import sys
'''
[문제 설명 및 알고리즘]
ABBA (O), ABAB(X), 위로 아치형 곡선을 그렷을 때, CROSS -> X
if(스택이 비어있거나 스택[-1] 와 현재 인덱스 비교해서 일치 X) push
else(스택이 비어있거나 peek() == 현재 인덱스 이면) pop()
word 수만큼 다 돌고 나서 스택이 다 비어 있으면 좋은 단어
[놓친 부분]
pop() 이외에도 스택의 가장 상단 부분을 보는 peek()을 이용하자
'''
def calcul_word(words):
    cnt = 0
    for word in words:
        stack = []
        for char in word:
            if(len(stack) == 0 or stack[-1]!=char):
                stack.append(char)
            else:
                stack.pop()
        if(len(stack) == 0): cnt+=1
    print(cnt)

words = []
n = int(sys.stdin.readline())

for _ in range(n):
    word = sys.stdin.readline().strip()
    words.append(word)
#print(words)
calcul_word(words)