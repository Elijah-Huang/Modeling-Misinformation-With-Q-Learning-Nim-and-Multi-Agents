from Nim.agents import*

for heaps in range(1, 4):
    print(f'Heaps: {heaps}')
    print('Random: ', RandomAgent().test_accuracy_random(heaps, 5, 10**6))
    print('Reverse Optimal: ', ReverseOptimalAgent().test_accuracy_random(heaps, 5, 10**6))
    print('Random Reverse Optimal: ', RandomReverseOptimalAgent().test_accuracy_random(heaps, 5, 10**6))
    print()

# OUTPUT:
#
# Heaps: 1
# Random:  0.456723 ~ 0.46
# Reverse Optimal:  0.199538 ~ 0.2
# Random Reverse Optimal:  0.199868 ~ 0.2
#
# Heaps: 2
# Random:  0.383212 ~ 0.38
# Reverse Optimal:  0.199956 ~ 0.2
# Random Reverse Optimal:  0.200052 ~ 0.2
#
# Heaps: 3
# Random:  0.338323 ~ 0.34
# Reverse Optimal:  0.168008 ~ 0.17
# Random Reverse Optimal:  0.167685 ~ 0.17
