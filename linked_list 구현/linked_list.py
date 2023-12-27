#클래스로써 노드를 정의한다.
class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next 

#연결리스트를 만들고 node1~node4까지의 포인터 구성
def init() :
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node4 = Node(4)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    return node1 #head 노드를 반환

#연결 리스트에서 해당 데이터를 지우고 나머지를 연결
def delete(head, del_data):
    pre_node = head
    next_node = pre_node.next

    #삭제할 데이터가 head 노드이면, head를 옮겨준뒤, 삭제, 함수 종료 
    if pre_node.data == del_data:
        head = next_node
        del pre_node
        return 
    
    #한칸 한칸 이동하면서 삭제할 노드 탐색
    while next_node: #next_node가 None이 아닐때까지 
        if(next_node.data == del_data):
            pre_node.next = next_node.next
            del next_node
            break #while 문 빠져나옴 
        #한칸 옮기는 과정 (pre 먼저 옮겨주고 next 이동)
        pre_node = next_node
        next_node = next_node.next
    return head
    
def pre_insert(head, ins_data):
    new_node = Node(ins_data)
    new_node.next = head
    head = new_node #head 노드 재설정 
    return head

#코드 추가 -> 리스트의 가장 끝에 삽입 
def back_insert(head, ins_data):
    new_node = Node(ins_data) #노드 객체 생성

    tmp_node = head
    # *내 다음 노드의 포인터가 None이 아니면 True*
    # 즉, 리스트의 끝까지 tmp_node 를 이동시킨다.
    while(tmp_node.next!=None):
        tmp_node = tmp_node.next
    tmp_node.next = new_node

#코드 추가 -> 리스트의 해당 원소 다음에 삽입 (중간 삽입)
def middle_insert(head, pre_data, ins_data):
    new_node = Node(ins_data)
    tmp_node = head

    while(tmp_node):
        if(pre_data == tmp_node.data):
            new_node.next = tmp_node.next 
            tmp_node.next = new_node
            break
        tmp_node = tmp_node.next 

def print_list(head):
    tmp = head
    while tmp: #tmp가 null이 아닐 때까지
        print(tmp.data)
        tmp = tmp.next 
    print() #개행

def LinkedList():
    head = init()
    print("Before : ")
    print_list(head)
    head = delete(head, 2)
    head = pre_insert(head, 8)
    middle_insert(head, 1, 13)
    back_insert(head, 6)
    print("After : ")
    print_list(head)

LinkedList()