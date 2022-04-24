from random import randint


# What is Reinforcement Learning?
# Similar to how humans learn
#
# Agent performs an action (on the)-> Environment (gives)-> Feedback to the agent (current state + a reward).
# The agent learns to make those actions if it gets a positive reward
# , but not to make them if it gets a negative reward (reinforcement learning).
# After many of these cycles, the agent learns how to optimally act on the environment for each state


# What is Nim?
#   2 player game where players take turns making moves (all impartial games are like this)
#
#   The game starts with an arbitrary number of heaps. Each heap has an arbitrary number of elements
#
#   In one move:
#     a player selects any heap (with a positive number of stones)
#     the player selects any positive number of elements and remove them from the heap
#
# Example of a game of Nim:
#   3 heaps
#   Starting heaps: 10, 2, 1
#
#   The first move: player 1 selects the 1st heap and remove 5 elements
#   Heaps after the first move: 5, 2, 1
#   The second move: player 2 selects the 2nd heap and removes all 2 elements
#   Heaps after the second move: 5, 0, 1
#   The third move: player 1 selects the 1st heap and remove 3 elements
#   Heaps after the third move: 2, 0, 1
#   ...
#   ...
#   1st player makes a move:
#   After that move: 0, 0, 0
#
#   player 1 wins the game and player 2 loses


def generate_heaps(num_heaps, max_element):
    heaps = [randint(1, max_element) for i in range(num_heaps)]
    heaps.sort(reverse=True)

    return heaps


def play_game(heaps, agent1, agent2, store_game=0):
    full_game = []

    turn = 0
    while heaps:
        if store_game:
            full_game.append(heaps.copy())

        turn += 1

        # find action that the current agent moving wants to do
        if turn & 1:
            heap_chosen, elements_to_remove = agent1.action(heaps)
        else:
            heap_chosen, elements_to_remove = agent2.action(heaps)

        # perform action
        heaps[heap_chosen] -= elements_to_remove

        # maintain sorted order of heap in O(N)
        if heaps[heap_chosen] == 0:
            heaps.pop(heap_chosen)  # remove heap from heaps since it is empty
        else:
            # keep swapping heap_chosen with heap_chosen+1 until heaps is sorted
            while heap_chosen < len(heaps) - 1 and heaps[heap_chosen] < heaps[heap_chosen + 1]:
                heaps[heap_chosen], heaps[heap_chosen + 1] = heaps[heap_chosen + 1], heaps[heap_chosen]
                heap_chosen += 1
    if store_game:
        full_game.append(heaps.copy())

    # turn is now the total turns in the completed game

    win = turn & 1  # 1 if agent 1 won else 0 if agent 2 won

    return win, full_game


class Random_agent:
    def action(self, heaps):
        heap_chosen = randint(0, len(heaps) - 1)
        elements_to_remove = randint(1, heaps[heap_chosen])

        return heap_chosen, elements_to_remove


class Optimal_agent:
    def action(self, heaps):
        heap_chosen, elements_to_remove = None, None  # returns

        xor_sum = 0  # xor sum of elements in heaps
        for elements in heaps:
            xor_sum ^= elements

        if xor_sum == 0:
            # no matter what move the agent does, it will not be optimal
            # thus, the agent can select any move, so we pick a move that takes O(1) to perform
            heap_chosen = len(heaps) - 1
            elements_to_remove = 1
        else:
            max_set_bit = 0  # max set bit in xor_sum
            while 1 << max_set_bit + 1 <= xor_sum:
                max_set_bit += 1

            # find heap with mx_set_bit set, which is guaranteed to exist
            for heap in range(len(heaps)):
                if (heaps[heap] >> max_set_bit) & 1:
                    heap_chosen = heap

            new_elements = xor_sum ^ heaps[heap_chosen]  # guaranteed to be < heaps[heap_chosen]
            elements_to_remove = heaps[heap_chosen] - new_elements

        return heap_chosen, elements_to_remove


class Q_learning_agent:
    def action(self, heaps):
        return heap_chosen, elements_to_remove


def test_optimal_agent(num_games=10 ** 4, max_heaps=100, max_elements=100):
    for game in range(1, num_games + 1):
        heaps = generate_heaps(max_heaps, max_elements)

        win, full_game = play_game(heaps, Optimal_agent(), Random_agent(), 1)

        # although the xor sum of the intial heaps might not be 0,
        # all the optimal agent needs to win is for the random agent to play
        # a non-optimal move (not make xor sum = 0), which is almost guaranteed for the game size
        # the first agent should lose
        if win == 0:
            print(f"Bad Game:")
            for heaps in full_game:
                print(heaps)


test_optimal_agent()  # does not print "Bad Game", so all games were won
