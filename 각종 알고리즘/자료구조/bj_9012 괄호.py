import sys
'''
[내가 생각한 방안 HOW?]
1. 리스트의 요소들을 모두 스택에 담는다.
2. '(' 가 아닐때까지 스택에서 빼낸다. (꺼낼때마다 카운트)
3. ')'가 아닐때까지 스택에서 뺀다. (다른 변수로 카운트)
** 2번 카운트 값이랑 3번 카운트 값이 상이하면 break, print("NO")
4. 스택이 비어있을때까지 POP 해서 다 빼냈으면 print("YES")
[놓친 부분]
단순히 (( )) 경우만 있는것이 아니라 (() ()) 케이스도 존재함
즉 중괄호안에 소괄호가 있는 형태를 간과함
'(' 문자면 스택에 push, ')' 문자이고, 스택이 비어있지 않으면 pop <'(' 괄호 한정>
*pop 할때 스택 비어있으면 False
while 다 돌고 나서 스택에 '(' 가 아직도 남아있다면 False, empty면 True
'''
def decision(li):
    stack = []
    for i in range(len(li)):
        if(li[i] == '('):
            stack.append(li[i])
        else:
            if(len(stack)!=0):
                stack.pop()
            else:
                return False
    if(len(stack) == 0):
        return True
    else:
        return False
n = int(sys.stdin.readline())

for i in range(n):
    if(decision(list(sys.stdin.readline().strip()))==True):
        print("YES")
    else:
        print("NO")