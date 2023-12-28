'''
버킷 정렬 : 자료를 버킷 단위의 공간에 인덱스 매핑하여 저장,
처음~마지막까지 루프 돌면서 값이 있다면 출력하는 알고리즘 
But, 단점 : data 범위에 따라 그만큼의 버킷 공간이 필요함
이를 보완한 것이 기수 정렬 : 각 자릿수별로 버킷 정렬을 반복적으로 실행한 것
[알고리즘]
1. 0~9까지의 2D 인덱스 리스트를 만든다.
2. (num/position)%10을 통해 자릿수를 fetch, 값을 해당 인덱스 리스트에 할당
3. digit_number > 0을 통해 가장 큰 정수의 자릿수만큼 훑었는지 확인
4. 인덱스 0초기화, order 리스트 갱신 
5. position*=10을 통해 자릿수 증가 
[시간 복잡도]
시간 복잡도 : O(d * (n + b))
-> d는 정렬할 숫자의 자릿수, b는 10, n은 정렬할 원소의 개수
'''
def radix(order):
    isSorted = False
    position = 1

    while not isSorted: #바깥루프 
        isSorted = True
        queue_list = [list() for _ in range(10)]

        #이 루프에서 매 자릿수마다 버킷 정렬 -> 루프 break 없음
        for num in order:
            digit = (int) (num / position) % 10 #해당 자릿수를 뽑아낸다.
            queue_list[digit].append(num) #해당 인덱스 리스트에 num 할당
            #digit이 0이거나 0 미만일때(소숫점), 즉 자릿수 넘치면 루프 끝
            #한번 isSorted가 False가 되면 조건문을 더 이상 만족하지 않는다
            #->자릿수가 가장 긴 정수까지 고려한다는 의미 
            if(isSorted and digit > 0): #digit > 0은 아직 자릿수가 더 남았음을 의미 
                isSorted = False        

        #order 리스트 갱신하는 루프
        index = 0
        for numbers in queue_list:
            for num in numbers:
                order[index] = num
                index += 1
        #1 -> 10  -> 100 .. 자릿수 증가 역할 
        position *= 10

#x = [5, 2, 8, 6, 1, 9, 3, 7]
x = [12, 36, 128, 4456, 110, 5]
radix(x)
print(x)