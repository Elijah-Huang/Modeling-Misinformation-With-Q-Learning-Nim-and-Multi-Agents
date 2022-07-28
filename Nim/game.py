# Note that the Game class needs the Agent class from module agents for type hints, but agents needs the Game class
# To solve this circular dependency, we use "forward references": https://peps.python.org/pep-0484/#forward-references
import random
from typing import List


class Game:
    def __init__(self, num_heaps: int, max_elements: int, heaps : List[int] = None) -> None:
        """
        **Description:**
            - Generates a random game of nim (represented by heaps).
              Allows for easy action by action play of the game.

            - Computationally efficient.
              The heaps are maintained in sorted order to reduce the number of states by ~heaps!

        **Parameters:**
            - num_heaps: int

              + maximum number of heaps

            - max_elements: int

              + maximum number of elements per heap

            - heaps

              + initial game state (leave argument empty to create a random game)

        **Attributes:**
            - num_heaps: int

              + number of heaps in initial game (only used to generate random game)

            - max_elements: int

              + maximum number of elements per heap (only used to generate random game)

            - heaps: List[int]

              + current state represented by heaps of elements

              + the heaps are maintained in reverse sorted order and all empty heaps are removed (no zeros)

              + maintained in O(self.num_heaps) per operation

            - xor_sum: int

              + xor_sum of self.heaps
        """

        self.num_heaps = num_heaps
        self.max_element = max_elements
        if heaps:
            self.heaps = heaps.copy()
        else:
            self.heaps = sorted([random.randint(1, max_elements)
                                 for i in range(random.randint(1, num_heaps))], reverse=True)
        self.xor_sum = 0
        for heap in self.heaps:
            self.xor_sum ^= heap

    def perform_action(self, heap_chosen: int, elements_to_remove: int) -> bool:
        """
        **Description:**
          - Performs action on self.heaps (state).

          - Maintains sorted order of heaps in O(num_heaps).

        **Return:**
             - True if the action was optimal else False
        """

        try:
            prev_xor_sum = self.xor_sum
            self.xor_sum ^= self.heaps[heap_chosen]
            self.heaps[heap_chosen] -= elements_to_remove
            self.xor_sum ^= self.heaps[heap_chosen]

            # maintain sorted order of heaps in O(num_heaps)
            if self.heaps[heap_chosen] == 0:
                self.heaps.pop(heap_chosen)  # remove heap from heaps since it is empty
            else:
                # keep swapping heap_chosen with heap_chosen+1 until heaps is sorted
                while heap_chosen < len(self.heaps) - 1 and self.heaps[heap_chosen] < self.heaps[heap_chosen + 1]:
                    self.heaps[heap_chosen], self.heaps[heap_chosen + 1] = self.heaps[heap_chosen + 1], self.heaps[heap_chosen]
                    heap_chosen += 1

            # see if action was optimal
            # Note that any move is optimal if the xor is zero, but this occurs with
            # low probability (~1/elements because it's expected to be a random number)
            if prev_xor_sum == 0:
                # any action is optimal
                return True
            else:
                if self.xor_sum == 0:
                    return True
                else:
                    return False
        except:
            raise Exception("Invalid Action")

    def finished(self) -> bool:
        """
        **Return:**
            - Returns if the game is finished

              + The game is finished if all heaps are empty
        """

        return len(self.heaps) == 0

    def play_game(self, agent1 : 'Agent', agent2 : 'Agent', store_game : bool = False) -> tuple:
        """
        **Parameters:**
            - agent1: Agent

              + agent that moves first

            - agent2: Agent

              + agent that moves second

            - store_game: bool

              + if we store the game or not

        **Return:**
            - 1 if agent 1 won else 0, the full game
        """

        full_game = []

        turn = 0
        while not self.finished():
            if store_game:
                full_game.append(self.heaps.copy())

            turn += 1

            # find action that the current agent moving wants to do
            if turn & 1:
                action = agent1.action(self.heaps)
            else:
                action = agent2.action(self.heaps)
            # perform action
            self.perform_action(*action)
        if store_game:
            full_game.append(self.heaps.copy()) # self.heaps should be an empty list

        # turn is now the total turns in the completed game

        win = turn & 1  # 1 if agent 1 won else 0 if agent 2 won

        return win, full_game
