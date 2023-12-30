import sys

def count_circle(str):
    #내가 받은 매개변수 str은 O,X를 담고 있는 리스트야.
    #여기서 O의 개수만 세고 싶어.
    #근데 O가 연속으로 나타나면 ++ 해줘야
    cnt = 0
    sum = 0 #sum이라는 변수를 하나 더 두어서 지금까지의 카운트를 더한다.
    for i in range(len(str)):
        if(str[i] == 'O'):  #'O' 표시면 카운트 증가
            cnt += 1 
            sum += cnt #sum에 2, 3, 4... 계속 더해줌
        else:
            cnt = 0 #연속적이지 않으면 다시 0 초기화됨
    print(sum)

n = int(sys.stdin.readline())

for _ in range(n):
    str = list(sys.stdin.readline())
    str = str[:-1]
    count_circle(str)