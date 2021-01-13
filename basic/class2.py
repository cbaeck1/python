'''
왜 우리는 OOP를 사용해야 할까?
모듈화, 재사용성, 가독성, 전문화, 관리의 용이함 

<요구사항>
교사의 주요 업무 중 하나는 학생들의 성적을 확인하고 관리
1. 관리하는 반 : 3반
2. 학생 : 5명, 가현, 영희, 모모, 정석, 철수
3. 시험과목 : 국어, 영어, 수학

<기능>
1. 학급의 과목별로 평균 구하기
2. 학생 개개인 별로 성적 구하기
3. 학생 추가하기

1. 자료구조 : list
1-1. 첫번째 원소 : 반이름:  str
1-2. 두번째 원소 : 학생들: dict 
gradebook = ['3반'] 
students_1 = {'name': '가현', 'Math': 90, 'English': 70, 'Korean': 100}
students_2 = {'name': '영희', 'Math': 40, 'English': 65, 'Korean': 70}
students_3 = {'name': '모모', 'Math': 89, 'English': 82, 'Korean': 92}
students_4 = {'name': '정석', 'Math': 50, 'English': 62, 'Korean': 31}
students_5 = {'name': '철수', 'Math': 88, 'English': 20, 'Korean': 60}

2. 기능을 함수로 구현하기


3. 함수의 문제점
3-1. 모든 함수와 변수가 전역(global) 처리되어 있다.
스코프는(namespace라고도 한다.) 정말 꼭 필요한 경우가 아니면 범위를 한정짓는 게 맞다.
예기치 못한 오염의 위험이 있으며, 모든 변수가 한 스코프에 있기 때문에 관리하기도 힘들다.
함수안에서 유지할 변수와 전역적으로 유지할 변수를 구분하여야 한다.
3-2. 코드 가독성이 떨어진다.
자료가 3차원 이상이 되면서 원하는 함수를 만드는데 코드에 매직넘버가 들어가는 등 어려워졌다. 
아까 정의한 새 gradebook 만 봐도 가슴이 편치 않음을 느낄 수 있다. 
애초에 자료구조가 3차원 이상으로 넘어가기 시작하면, 클래스 사용을 진지하게 고려해야 한다.
3-3. 전문화가 어렵다.
모든 변수와 함수가 같은 스코프에 있기 때문에 가령 학생 전문 함수를 만들기가 더 어렵다.
아마 새 함수가 기존의 학생 함수들을 쓸 수도 있는데(의존할 수도 있는데) 함수들이 너무 많아 학생부 변수들도 봐야 한다.
3-4. 재사용성이 떨어진다.
지금은 한 파일에 학생과 관련된 정보, 학생부와 관련된 정보가 혼잡하게 섞여 있어 원하는 정보를 찾기 어렵다.
만약 동료 교사가 이것을 사용하려 하면, 원하는 기능 하나를 찾기 위해 이리 저리 함수를 돌아다녀야 한다.
커스터마이제이션이 어려워진다. 가령 기존의 함수에서 조금씩을 바꿔 쓰고자 한다고 하자.

4. 클래스 구현
<추가 데이터>
4. 시험종류 : 전국연합학력평가 3, 4, 7, 10월 정보
5. 학생들 취미
6. 반 추가

<추가 기능>
4. 학생들이 가장 좋아하는 취미가 뭔지 출력하기

특정 학생부와 학생이 자신 고유의 범위 안에 있다. 따라서 오염 등의 문제가 없다.
아까와 비교해보자. 아까는 add_student(gradebook, name, math, korean, english):처럼 함수가 학생부를 받았다. 
하지만 이번에는 gradebook.add_student(student)처럼 **학생부가 자신의 함수를 호출해 쓴다. 
오염의 문제가 없을 뿐만 아니라, 함수 실행 시 잘못 된 입력을 넣을 가능성도 줄어든다.
전문화가 보다 쉽다.
코드가 학생과 학생부로 나뉘어 좀더 그 클래스에 특화된 함수를 만들 수 있다. 그러니까 코드의 책임영역이 나뉘었다.
재사용성이 좋다.(가독성이 좋다.)
사용할 때 더 쉽지 않은가? 가독성도 더 좋다고 말해달라.
customization이 매우 쉬워졌다.
Base 클래스를 만들어 좀더 초반부터 닦고 싶은 사람은 이 클래스를 상속 받아 기능을 구현할 수 있고,
누구는 SunghwanGradebook를 그대로 가져다 쓸 수도 있으며,
또 SunghwanGradebook를 상속해서 쓸 수도 있을 것이다. 이렇게 되면
이름이 충돌해 문제가 생길리 없다.
문서화가 더 쉬워졌다.
언제 어디서나 문서화는 좋은 관리 방안이다.
클래스 선언 바로 밑에 파이썬에서 docstring이라고 부르는 문자열을 추가했다.
그러면 help(SunghwanGradebook)과 같이 쓰면 그 문자열이 출력되어
사용시 유용한 문자열을 확인할 수 있다.
그러니까 클래스에는 다루는 클래스에 대한 무수한 정보가 있지만,
클래스를 감싸고 자신의 메소드라는 인터페이스만 남기고,
이를 문서화해 사용하는 사람은 클래스에 대한 무수하고 잡스러운 정보를 기억할 필요 없이
클래스를 사용할 수 있게 되었다.

'''

classGradebook = ['3반'] 
students_1 = {'studentName': '가현', 'Math': 90, 'English': 70, 'Korean': 100}
students_2 = {'studentName': '영희', 'Math': 40, 'English': 65, 'Korean': 70}
students_3 = {'studentName': '모모', 'Math': 89, 'English': 82, 'Korean': 92}
students_4 = {'studentName': '정석', 'Math': 50, 'English': 62, 'Korean': 31}
students_5 = {'studentName': '철수', 'Math': 88, 'English': 20, 'Korean': 60}

classGradebook.append(students_1)
classGradebook.append(students_2)
classGradebook.append(students_3)
classGradebook.append(students_4)
classGradebook.append(students_5)

### 1. 학급의 과목별로 평균 구하기
def grade_average(classGradebook, subject):
    className, gradebook = classGradebook[0], classGradebook[1:]  # packing이 들어갔다.
    n = len(gradebook)  # 학생수 구하기
    average = sum([x[subject] for x in gradebook])  / n 
    return className + '의 ' + subject + ' 평균은 ' + str(average)
    
print(grade_average(classGradebook, 'Math'))
print(grade_average(classGradebook, 'Korean'))
print(grade_average(classGradebook, 'English'))
# 3반의 Math 평균은 71.4
# 3반의 Korean 평균은 70.6
# 3반의 English 평균은 59.8

### 2. 학생 개개인 별로 평균 성적 구하기
def student_average(classGradebook, name):
    for stu in classGradebook[1:]:
        if stu['studentName'] == name:
            student = stu
            break
    math = student['Math']
    korean = student['Korean']
    english = student['English']
    avg = (math + korean + english) / 3
    return name + ' 평균은 ' + str(avg)

print(student_average(classGradebook, '가현'))
print(student_average(classGradebook, '영희'))
print(student_average(classGradebook, '모모'))
print(student_average(classGradebook, '정석'))
print(student_average(classGradebook, '철수'))

### 3. 학생 추가하기
def add_student(classGradebook, name, math, korean, english):
    newbie = {'studentName': name, 'Math': math, 'Korean': korean, 'English': english}
    classGradebook.append(newbie)
    
add_student(classGradebook, '사나', 50, 60, 70)

#############################################################################
class BaseGradebook:
    def __init__(self, className, teacherName):
        self.className = className
        self.teacherName = teacherName
        self.students = []        
        self.length = 0        
    
    # classGradebook['모모']와 같이 이름을 키로 받았을 때 특정 학생 반환하도록 반환
    def __get__(self, studentName):
        for stu in self.students:
            if stu.name == studentName:
                return stu

    # 특정 학생의 특정 시험 점수를 반환
    def get_specific_score(studentName, subject, which_exam):
        return self.students[studentName][subject][which_exam]

    # 학생 추가하기
    def add_student(self, student):
        self.students.append(student)
        self.length += 1

    # 학급의 특정 시험의 특정 과목 평균 출력 : 학생,과목,시험종류를 3차원으로
    def average_of_subject_which_exam(self, subject, which_exam):
        tmp_sum = 0
        for stu in self.students:
            tmp_sum += stu[subject][which_exam]
        return tmp_sum / self.length    

class BaseStudent:
    def __init__(self, studentName, sex):
        self._name = studentName
        self._sex = sex
        self._math = []  
        self._korean = []
        self._english = []
        self._hobbies = []
        
    def add_exam_result(self, math, korean, english):
        self._math.extend(math)
        self._korean.extend(korean)
        self._english.extend(english)
        #raise NotImplementedError("")

    # student['math'] 처럼 썼을 때 해당 과목 성적을 담은 리스트를 반환
    def __getitem__(self, subject):
        return getattr(self, '_' + subject)

    # 학생의 특정 과목 점수의 평균 반환
    def average_grade_of(self, subject):
        return self._name + '의 ' + subject + ' 평균은 ' + str(sum(self[subject]) / len(self[subject]))

    def fill_hobbies(self, hobby):
        self._hobbies.append(hobby)
        #raise NotImplementedError("")

    def print_hobby(self):
        return ', '.join(self._hobbies)
        #raise NotImplementedError("")
        
test_student = BaseStudent('나는', 'F')
print(test_student)
print(test_student._math)


# gradebook = ['3반',
#  {'English': [45, 75, 67, 70], 'Korean': [100, 97, 97, 100], 'Math': [65, 74, 87, 54], 'name': '가현',
#    'hobbies': ['dance', 'sing'], 'favorite_movies': ['Begin again', 'Chicken run']},
#  {'English': [54, 65, 75, 65], 'Korean': [70, 87, 45, 76], 'Math': [40, 54, 65, 65], 'name': '영희',
#    'hobbies': ['read', 'dringking beer'], 'favorite_movies': ['mugando', 'sinsegye', 'money ball']},
#  {'English': [82, 93, 95, 95], 'Korean': [67, 76, 78, 82], 'Math': [67, 87, 65, 87], 'name': '모모',
#    'hobbies': ['dance', 'Programming'], 'favorite_movies': ['Zootopia', 'Search', '99 homes']},
#  {'English': [87, 56, 54, 76], 'Korean': [54, 54, 67, 76], 'Math': [34, 33, 54, 45], 'name': '정석',
#    'hobbies': ['trip'], 'favorite_movies': ['Mother']},
#  {'English': [82, 81, 61, 79], 'Korean': [72, 88, 78, 98], 'Math': [67, 78, 33, 45], 'name': '철수',
#    'hobbies': ['Movie'], 'favorite_movies': ['Your name...']},
# ]
# print(student_average(classGradebook, '가현'))
# print(student_average(classGradebook, '영희'))
# print(student_average(classGradebook, '모모'))
# print(student_average(classGradebook, '정석'))
# print(student_average(classGradebook, '철수'))

gradebook = BaseGradebook('3반', '담임')
stu1 = BaseStudent('가현', 'F')
stu1.add_exam_result([45, 75, 67, 70], [100, 97, 97, 100], [65, 74, 87, 54])
gradebook.add_student(stu1)
stu2 = BaseStudent('영희', 'F')
stu2.add_exam_result([54, 65, 75, 65], [70, 87, 45, 76], [40, 54, 65, 65])
gradebook.add_student(stu2)
stu3 = BaseStudent('모모', 'F')
stu3.add_exam_result([82, 93, 95, 95], [67, 76, 78, 82], [67, 87, 65, 87])
gradebook.add_student(stu3)
stu4 = BaseStudent('정석', 'F')
stu4.add_exam_result([87, 56, 54, 76], [54, 54, 67, 76], [34, 33, 54, 45])
gradebook.add_student(stu4)
stu5 = BaseStudent('철수', 'F')
stu5.add_exam_result([82, 81, 61, 79], [72, 88, 78, 98], [67, 78, 33, 45])
gradebook.add_student(stu5)

print(stu1.average_grade_of('math'))
print(gradebook.average_of_subject_which_exam('math', 2))
print(gradebook)


