import sys

n = int(sys.stdin.readline())

stack = []
for _ in range(n):
    order = list(map(int, sys.stdin.readline().split()))
    if(order[0] == 1):
        val = order[1]
        stack.append(val)
    elif(order[0] == 2):
        if(len(stack) == 0):
            print(-1)
        else:
           print(stack.pop())
    elif(order[0] == 3):
        print(len(stack))
    elif(order[0] == 4):
        if(len(stack) == 0):
            print(1)
        else:
            print(0)
    else:
        if(len(stack) != 0):
            print(stack[-1])
        else:
            print(-1)