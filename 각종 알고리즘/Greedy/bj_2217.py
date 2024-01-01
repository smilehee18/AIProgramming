import sys
'''
[알고리즘]
1. 입력받은 로프들의 중량들을 오름차순으로 sorting(큰수부터 pop하기 위해)
2. 로프 병렬로 연결 시 예를 들어 10, 15 로프 있을 때,
* 15가 더 들 수 있다고 하더라도, 10인 로프는 10이상이면 끊어짐 *
3. 따라서 로프 연결시 들 수 있는 물건의 최대 중량 W = min(로프 중량)*로프 개수
4. 중량 리스트에서 하나씩 꺼낸 수랑 개수 곱하는 식으로 max_weight 갱신
[놓친 부분]
1. i번째 루프마다 case 리스트를 통해 해결하려고 했으나 메모리 초과
2. 코드 최적화를 위해 max_weight를 갱신하는 방식으로 수정
'''
n = int(sys.stdin.readline())

lope = []
max_weight = 0
for _ in range(n):
    lope.append(int(sys.stdin.readline()))

lope.sort() #로프들의 중량 오름차순으로 정렬
# print(lope)

for i in range(n):
    weight = lope.pop() #i개 로프 중량 fetch
    max_weight = max(max_weight, weight*(i+1))
print(max_weight)