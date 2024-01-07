from collections import deque
import sys

n = int(sys.stdin.readline())

d = deque()
for i in range(n):
    str = sys.stdin.readline().split()
    order = str[0]
    if(order == "push_front"):
        val = str[1]
        d.appendleft(val)
    elif(order == "push_back"):
        val = str[1]
        d.append(val)
    elif(order == "size"):
        print(len(d))
    elif(order == "front"):
        if(len(d)==0):
            print(-1)
        else:
            print(d[0])
    elif(order == "back"):
        if(len(d)==0):
            print(-1)
        else:
            print(d[-1])
    elif(order == "empty"):
        if(len(d)==0):
            print(1)
        else:
            print(0)
    elif(order == "pop_front"):
        if(len(d)==0):
            print(-1)
        else:
            num = d.popleft()
            print(num)
    elif(order == "pop_back"):
        if(len(d)==0):
            print(-1)
        else:
            num = d.pop()
            print(num)
