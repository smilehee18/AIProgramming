import sys

m, n = map(int, sys.stdin.readline().split())

dict = {'1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six',
        '7':'seven', '8':'eight', '9':'nine', '0':'zero'}

li = []

for i in range(m, n+1):
    a = ' '.join([dict[j] for j in str(i)])
    li.append([i, a]) #[8, 'eight'], [9, 'nine'], [10, 'one zero'] 와 같은 값들이 저장됨

#print(li) 
li.sort(key=lambda x:x[1]) #문자열 기준으로 정렬시킴

for n in range(len(li)):
    if(n%10==0 and n!=0): #10개씩마다 개행하기 위해서
        print()
    print(li[n][0], end=' ')