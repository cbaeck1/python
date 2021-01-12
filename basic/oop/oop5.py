# -*- coding: utf-8 -*-

# 5. 상속과 서브 클래스(Inheritance and Subclass)

class Unit(object):
    def __init__(self, rank, size, life):
        self.name = self.__class__.__name__
        self.rank = rank
        self.size = size
        self.life = life
        
    def show_status(self):
        print('이름: {}'.format(self.name))
        print('등급: {}'.format(self.rank))
        print('사이즈: {}'.format(self.size))
        print('라이프: {}'.format(self.life))
        

class Goblin(Unit):
    pass

goblin_1 = Goblin('병사', 'Small', 100)

goblin_1.show_status()