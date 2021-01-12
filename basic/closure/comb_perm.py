#

import itertools

my_list = [1,2,3]

combinations = itertools.combinations(my_list, 2)
print("combinations")
for c in combinations:
    print(c)

permutations = itertools.permutations(my_list, 2)
print("permutations")
for p in permutations:
    print(p)

#
my_list2 = [1,2,3,4,5,6]

combinations = itertools.combinations(my_list2, 3)
permutations = itertools.permutations(my_list2, 3)
print("combinations : sum(result) == 10")
print([result for result in combinations if sum(result) == 10])

#
word = 'sample'
my_letters = 'plmeas'

combinations = itertools.combinations(my_letters, 6)
permutations = itertools.permutations(my_letters, 6)

print("combinations")
for p in combinations:
    print(p)
    if ''.join(p) == word:
        print('Match!')
        break
else:
    print('No Match!')

print("permutations")
for p in permutations:
    #print(p)
    if ''.join(p) == word:
        print('Match!')
        break
else:
    print('No Match!')