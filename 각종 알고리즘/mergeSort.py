'''
mergeSort 알고리즘 
핵심 : 반띵의 반띵을 반복해서 원소가 1개일때까지 쪼갠다음에,
왼쪽, 오른쪽 리스트 원소 하나하나씩 비교해가면서
더 작은 애를 최종 리스트에 넣어간다. 
언제까지? 왼 오 리스트 중에 하나라도 크기가 넘어갈 때까지,
남은 원소는 마지막에 채워준다. 
Time Memory Trade Off (TMTO)
시간복잡도 : O(nlogn) 
But, 임시 리스트 2개가 필요하기 때문에 메모리 사용량 증가
'''
def mergeSort(x):
    if len(x) > 1: #반띵을 얼마나 할거니? 조건 -> 원소 개수가 1개일때까지
        mid = len(x)//2
        colx, rowx = x[:mid], x[mid:] #반띵해서 리스트로 
        print(f'rowx',rowx); print(f'colx',colx)
        mergeSort(colx) #리스트를 매개변수로 주어서 재귀 호출
        mergeSort(rowx) #리스트 원소가 1개 일때까지 호출 -> 거꾸로 리턴

        print("here")
        coli, rowi, i = 0, 0, 0
        print(f'Arowx',rowx); print(f'Acolx',colx)
        while coli < len(colx) and rowi < len(rowx):
            if(colx[coli] < rowx[rowi]):
                x[i] = colx[coli]
                coli += 1
            else:
                x[i] = rowx[rowi]
                rowi += 1
            i+=1
        x[i:] = colx[coli:] if coli != len(colx) else rowx[rowi:]
        print(f'x', x)

x = [44, 3, 252, 17, 6, 93]
print("Before : ")
print(x)
mergeSort(x)
print("After Merge-Sort: ")
print(x)