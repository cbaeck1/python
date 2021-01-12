
it = iter([1, 2, 3])  # [1, 2, 3]의 반복자 구하기
# 반복자: next() 함수로 값을 하나씩 꺼낼 수 있는 데이터
# iter() 함수: 반복 가능한 데이터를 입력받아 반복자를 반환하는 함수
# next() 함수: 반복자를 입력받아 다음 출력값을 반환하는 함수
print(it)

print(next(it))
print(next(it))
print(next(it))

# Exception has occurred: StopIteration
# print(next(it))
