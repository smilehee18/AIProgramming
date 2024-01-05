import sys
'''
[How?] 이진 탐색 이용하여 해당 수가 있다면 True, 없으면 False리턴
이진 탐색 : 탐색 범위를 오름차순으로 정렬한 뒤,
low, high 인덱스를 지정, 내가 찾고자 하는 val 값과 일치 여부 확인
**만약, val값이 mid값보다 크다면, low = mid+1로 설정
**만약, val값이 mid보다 작다면, high = mid-1로 설정
핵심은 탐색 범위를 1/2로 줄여나감으로써 효율적인 탐색을 구현한다.
시간 복잡도 : 최선의 경우 -> 한번에 탐색 완료 O(1)
반띵의 반띵을 반복해서 가장 나중에 찾은 경우 (최악) : O(log2n)
이진 탐색은 시간 제한 조건이 있을때 활용하기 매우 좋은 알고리즘!!!
'''

def find_val(n, val, nlist):
    low = 0
    high = n-1
    while(low <= high):
        mid = (low+high)//2
        if(val == nlist[mid]):
            return True
        elif(val > nlist[mid]):
            low = mid+1
        else:
            high = mid-1
    return False

n = int(sys.stdin.readline())
nlist = list(map(int, sys.stdin.readline().split()))
m = int(sys.stdin.readline())
mlist = list(map(int, sys.stdin.readline().split()))

nlist.sort()

for i in range(m):
    val = mlist[i]
    if(find_val(n, val, nlist) == True):
        print(1)
    else:
        print(0)