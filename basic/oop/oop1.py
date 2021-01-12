# -*- coding: utf-8 -*-


# 1. 객체 지향 프로그래밍(OOP) 왜 사용하는가?
# 게임의 캐릭터를 만드는 예제
#   속성 : 이름, 에너지, 데미지, 인벤토리 

###############################################################
# 1. 동일한 속성이 반복적으로 발생 : 중복된 코드
# 히어로 1
hero_1_name = '아이언맨'
hero_1_health = 100
hero_1_damage = 200
hero_1_inventory = [
    {'gold': 500},
    {'weapon': '레이저'}
]

# 히어로 2
hero_2_name = '데드풀'
hero_2_health = 300
hero_2_damage = 30
hero_2_inventory = [
    {'gold': 300},
    {'weapon': '장검'}
]

# 히어로 3
hero_3_name = '울버린'
hero_3_health = 200
hero_3_damage = 50
hero_3_inventory = [
    {'gold': 350},
    {'weapon': '클로'}
]

# 몬스터 1
monster_1_name = '고블린'
monster_1_health = 90
monster_1_damage = 30
monster_1_inventory = [
    {'gold': 50},
    {'weapon': '창'}
]

# 몬스터 2
monster_2_name = '드래곤'
monster_2_health = 200
monster_2_damage = 80
monster_2_inventory = [
    {'gold': 200},
    {'weapon': '화염'}
]

# 몬스터 3
monster_3_name = '뱀파이어'
monster_3_health = 80
monster_3_damage = 120
monster_3_inventory = [
    {'gold': 1000},
    {'weapon': '최면술'}
]

###############################################################
# 2. 중복된 코드를 리스트를 사용하여 제거한 후 캐릭터의 선택은 인덱스로 
# "아이언맨"은 인덱스 0, "데드풀"은 인덱스 1, "울버린"은 인덱스 2를 사용하여 케릭터의 데이터에 엑세스
# 개발자 실수로 함수에 오류가 있음
import json

hero_name = ['아이언맨', '데드풀', '울버린']
hero_health = [100, 300, 200]
hero_damage = [200, 30, 50]
hero_inventory = [
    {'gold': 500,'weapon': '레이저'},
    {'gold': 300, 'weapon': '장검'},
    {'gold': 350, 'weapon': '클로'}
]

monster_name = ['고블린', '드래곤', '뱀파이어']
monster_health = [90, 200, 80]
monster_damage = [30, 80, 120]
monster_inventory = [
    {'gold': 50,'weapon': '창'},
    {'gold': 200, 'weapon': '화염'},
    {'gold': 1000, 'weapon': '최면술'}
]

# 히어로가 죽으면 호출되는 함수
# 히어로의 에너지가 0이 되어 죽었을때 히어로를 리스트에서 지우는 함수
def hero_dies(hero_index):
    del hero_name[hero_index]
    del hero_health[hero_index]
    del hero_damage[hero_index]
    # <--- 개발자가 실수로 del hero_inventory[hero_index]를 빠뜨렸네요.
    
hero_dies(0)

# 아이언맨 죽음
print(hero_name[0])
print(hero_health[0])
print(hero_damage[0])
print(json.dumps(hero_inventory[0], ensure_ascii=False))

###############################################################
# 3. 리스트를 사전으로 변경, 함수를 리스트 자료형 기능으로 대체하여 해결
# 키와 값의 반복
heroes = [
    {'name': '아이언맨', 'health': 100, 'damage': 200, 'inventory': {'gold': 500, 'weapon': '레이저'}},
    {'name': '데드풀', 'health': 300, 'damage': 30, 'inventory': {'gold': 300, 'weapon': '장검'}},
    {'name': '울버린', 'health': 200, 'damage': 50, 'inventory': {'gold': 350, 'weapon': '클로'}}
]

monsters = [
    {'name': '고블린', 'health': 90, 'damage': 30, 'inventory': {'gold': 50, 'weapon': '창'}},
    {'name': '드래곤', 'health': 200, 'damage': 80, 'inventory': {'gold': 200, 'weapon': '화염'}},
    {'name': '뱀파이어', 'health': 80, 'damage': 120, 'inventory': {'gold': 1000, 'weapon': '최면술'}}
]

print(json.dumps(heroes, ensure_ascii=False))
# 아이언맨 죽음 : 리스트의 0번째 자료
del(heroes[0])
print(json.dumps(heroes, ensure_ascii=False))

###############################################################
# 4. 키와 값의 반복 -> class 와 instance 로 (블루프린트)
# class 정의
class Character(object):
    def __init__(self, name, health, damage, inventory):
        self.name = name
        self.health = health
        self.damage = damage
        self.inventory = inventory
        
    def __repr__(self):
        return self.name
        
# Character 클래스의 오브젝트 생성
heroes = []
heroes.append(Character('아이언맨', 100, 200, {'gold': 500, 'weapon': '레이저'}))
heroes.append(Character('데드풀', 300, 30, {'gold': 300, 'weapon': '장검'}))
heroes.append(Character('울버린', 200, 50, {'gold': 350, 'weapon': '클로'}))

monsters = []
monsters.append(Character('고블린', 90, 30, {'gold': 50, 'weapon': '창'}))
monsters.append(Character('드래곤', 200, 80, {'gold': 200, 'weapon': '화염'}))
monsters.append(Character('뱀파이어', 80, 120, {'gold': 1000, 'weapon': '최면술'}))

print(heroes)  # 히어로 리스트 확인
print(monsters)  # 몬스터 리스트 확인
del(heroes[0])  # 히어로 리스트에서 아이언맨 삭제
print(heroes)  # 히어로 리스트 재확인

