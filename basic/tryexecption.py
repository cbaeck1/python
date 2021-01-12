'''
17. 예외 처리

1. 오류는 어떤 때 발생하는가?
  checked exception : 트랜잭션을 roll-back하지 않음
    SyntaxError: invalid syntax
  Runtime exception = Unchecked Exception, 트랜잭션을 roll-back
     ZeroDivisionError, NameError, TypeError, ValueError, OSError 
2. 오류 예외 처리 기법
2-1. try, except문
2-2. try .. finally
2-3. 여러개의 오류처리하기
3. 오류 회피하기
4. 오류 일부러 발생시키기
5. 예외 만들기




Error는 시스템 레벨의 심각한 수준의 에러 : 시스템에 변화를 주어 문제를 처리
Exception은 개발자가 로직을 추가하여 처리

'''