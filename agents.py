# note that importing agents also imports the Game class
import pickle
import random
import pandas as pd
import os
from matplotlib import pyplot as plt
from game import Game


class Agent:
    def action(self, game: Game) -> tuple:
        """
        **Description:**
          - Returns an action for the current game

        **Parameter:**
            - game: Game

              + current game

        **Return:**
            - a tuple representing the action: (heap chosen, elements to remove)
        """

        heap_chosen = None
        elements_to_remove = None
        return heap_chosen, elements_to_remove


class RandomAgent(Agent):
    def action(self, game: Game) -> tuple:
        """
        **Description:**
            - Returns a random action for the current game

        **Parameter:**
            - game: Game

              + current game

        **Return:**
            - a tuple representing the action: (heap chosen, elements to remove)
        """

        heap_chosen = random.randint(0, len(game.heaps) - 1)
        elements_to_remove = random.randint(1, game.heaps[heap_chosen])

        return heap_chosen, elements_to_remove


class OptimalAgent(Agent):
    def action(self, game: Game) -> tuple:
        """
        **Description:**
             - Find the optimal action for the current game via game theory

        **Parameter:**
            - game: Game

              + current game

        **Logic:**
            If the xor_sum of the heaps is 0:
                Any action is optimal (xor_sum becomes positive).
            Else:
                The optimal action changes the xor_sum to 0, and it always exists.

        **Return:**
            - a tuple representing the action: (heap chosen, elements to remove)
        """

        if game.xor_sum == 0:
            # no matter what move the agent does, it will not be optimal
            # thus, the agent can select any move, so we pick a move that takes O(1) to perform
            heap_chosen = len(game.heaps) - 1
            elements_to_remove = 1
        else:
            max_set_bit = 0  # max set bit in xor_sum
            while 1 << max_set_bit + 1 <= game.xor_sum:
                max_set_bit += 1

            # find heap with mx_set_bit set, which is guaranteed to exist
            for heap in range(len(game.heaps)):
                if (game.heaps[heap] >> max_set_bit) & 1:
                    heap_chosen = heap

            new_elements = game.xor_sum ^ game.heaps[heap_chosen]  # guaranteed to be < heaps[heap_chosen]
            elements_to_remove = game.heaps[heap_chosen] - new_elements

        return heap_chosen, elements_to_remove


class RandomReverseOptimalAgent(Agent):
    def action(self, game: Game) -> tuple:
        """
        **Description:**
            - Returns the worse possible action for the state "heaps"

              + Worse possible in terms of game theory and xor sums. However, the action space is not restricted like
                with ReverseOptimalAgent. Instead, the possible actions are tested in random order until a not optimal
                one is found. If only the optimal action is possible, then of course it is chosen.

        **Parameter:**
            - game: Game

              + current game

        **Return:**
            - a tuple representing the action: (heap chosen, elements to remove)
        """

        actions = []
        for heap in range(len(game.heaps)):
            for remove in range(1, game.heaps[heap]+1):
                actions.append((heap, remove))
        random.shuffle(actions)

        for heap, remove in actions:
            if game.xor_sum == 0:
                # any action is not optimal
                heap_chosen, elements_to_remove = heap, remove
                break
            else:
                # xor_sum != 0 after removal is not optimal
                if game.xor_sum ^ game.heaps[heap] ^ (game.heaps[heap]-remove) != 0:
                    heap_chosen, elements_to_remove = heap, remove
                    break
        else:
            # all actions were optimal (all heaps have 1 element I believe, but no need for potentially flawed casework)
            heap_chosen, elements_to_remove = actions[0]

        return heap_chosen, elements_to_remove


class ReverseOptimalAgent(Agent):
    def action(self, game: Game) -> tuple:
        """
        **Description:**
            - Returns the worse possible action for the state "heaps"

              + The only time the action is potentially optimal is if all heaps have 1 element

              + Not only is it the worse possible theoretically, but it also only removes 1 or 2 elements.
                This means the actions it takes are limited, preventing the amount that can be learned from them

        **Parameter:**
            - game: Game

              + current game

        **Return:**
            - a tuple representing the action: (heap chosen, elements to remove)
        """

        if game.heaps[0] == 1 or game.xor_sum ^ game.heaps[0] ^ (game.heaps[0]-1) != 0:
            # Either game.xor_sum == 0 -> second condition true, in this case any move we do is not optimal or
            # First condition of 'or' is true and all elements are 1, so we can only do this "same" move or
            # Second condition of 'or' is true and game.xor_sum != 0, so this move is not optimal
            heap_chosen = 0
            elements_to_remove = 1
        else:
            # Only possible if game.heaps[0] > 1, game.xor_sum != 0, and removing 1 element makes xor_sum = 0
            # Thus, removing 2 elements makes the xor_sum != 0, which is not optimal in this case
            heap_chosen = 0
            elements_to_remove = 2

        return heap_chosen, elements_to_remove


class QLearningAgent(Agent):
    def __init__(self, obj_dir: str, train_opponent_name: str = None,
                 num_heaps: int = None, max_elements: int = None,
                 alpha: float = None, gamma: float = None, epsilon: float = None,
                 reset: bool = False) -> None:
        """
        **Description:**
          - Initializes the Q Agent either by loading it from the file if it exists. Otherwise,
            initializes a fresh object and creates necessary directories and files

        **Parameters:**
            - obj_dir: str

              + directory where object data will be saved
                (object itself, csv of accuracy, plots of said csv)

            - train_opponent_name: str

              + Name of the opponent used to train the Q agent. Must be spelled exactly the same as the class name
                (make "self" for the agent to train by playing itself)

            - num_heaps: int

              + number of heaps in initial game

            - max_elements: int

              + maximum number of elements per heap

            - alpha: float

              + learning rate

            - gamma: float

              + discount factor

            - epsilon: float

              + exploration rate

            - reset: bool

              + if true, creates a new object from scratch instead of reloading from file (if file exists)

        **Attributes:**
            - obj_dir: str

              + directory where object data will be saved
                (object itself, csv of accuracy, plots of said csv).

            - train_opponent_name: str

              + name of the opponent used to train the Q agent
                (use self for the agent to train by playing itself).

            - train_opponent: Agent

              + object of some agent class determined by self.train_opponent_name

            - num_heaps: int

              + number of heaps in initial game.

            - max_elements: int

              + maximum number of elements per heap.

            - alpha: float

              + learning rate.

            - gamma: float

              + discount factor.

            - epsilon: float

              + exploration rate.

            - played_games: int

              + total number of games played during training (until last save point).

            - name: str

              + long name that describes Q agent based on parameters.

            - ascii_name: str

              + Same as self.name but using only ascii characters.

            - Q: Dict[tuple, Dict[tuple, float]]

              + Q table storing state-action pairs.
        """

        if not os.path.exists(obj_dir):
            os.mkdir(obj_dir)

        if not reset and os.path.isfile(obj_dir + '/obj') and os.path.exists(obj_dir + '/obj'):
            # load it
            with open(obj_dir + '/obj', "rb") as fin:
                obj = pickle.load(fin)
                self.__dict__.update(obj.__dict__)
        else:
            # we haven't created the file yet or reset=True -> we will make object from scratch

            # Remove all files in directory. This is dangerous if you accidentally put in the wrong directory, so
            # it only does this if the number of files == 3 (the amount it would be if used properly)
            if len(list(os.scandir(obj_dir))) == 3:
                for file in os.scandir(obj_dir):
                    os.remove(file.path)

            self.obj_dir = obj_dir
            self.train_opponent_name = train_opponent_name
            self.train_opponent = None
            self.set_train_opponent(train_opponent_name)
            self.num_heaps = num_heaps
            self.max_elements = max_elements
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.played_games = 0
            self.name = f'(Q Agent): train_opponent={self.train_opponent_name},\n' \
                        f'          num_heaps={self.num_heaps}, max_elements={self.max_elements},\n' \
                        f'          α={alpha}, γ={gamma}, ε={epsilon}'
            self.ascii_name = f'(Q Agent) train_opponent={self.train_opponent_name}, ' \
                              f'num_heaps={self.num_heaps}, max_elements={self.max_elements}, ' \
                              f'alpha={alpha}, gamma={gamma}, epsilon={epsilon}'

            # create new Q table
            self.Q = {}
            for i in range(pow(self.max_elements+1, self.num_heaps)):
                # i represents a state in base max_elements+1
                state = []
                while i:
                    state.append(i % (self.max_elements+1))
                    i //= self.max_elements + 1
                state.sort(reverse=True)
                while state and state[-1] == 0:
                    state.pop()

                if tuple(state) not in self.Q:
                    # add state and actions to Q
                    self.Q[tuple(state)] = {}

                    next_states = set()
                    for heap_chosen in range(len(state)):
                        for elements_to_remove in range(1, state[heap_chosen]+1):
                            game = Game(0, 0)
                            game.heaps = state.copy()
                            game.perform_action(heap_chosen, elements_to_remove)
                            new_state = tuple(game.heaps)

                            if new_state not in next_states:
                                # move leads to unique, new state
                                next_states.add(new_state)
                                self.Q[tuple(state)][(heap_chosen, elements_to_remove)] = 0 # initialize to 0

            # create text file who's name gives information about the agent (since the folder name is probably useless)
            with open(self.obj_dir + '/' + self.ascii_name, 'w') as fin:
                pass

            # create/save accuracy dataframe
            self.accuracy = pd.DataFrame(data=[[self.test_accuracy()]], index=[0], columns=[self.name])
            self.accuracy.index.rename('Games Played', inplace=True)
            self.accuracy.to_csv(self.obj_dir + '/accuracy.csv')

            self.save(self.obj_dir + '/obj') # save entire object

    def update_q(self, s: list, a: tuple, reward: float, sp: list = None) -> None:
        """
        **Description:**
            - Updates the Q table

        **Parameters:**
            - s: list

              + previous state

            - a: tuple

              + action at s

            - reward: float

              + If sp is None, (game is over): should be 1 if Q agent won else -1. Otherwise, 0.

            - sp: list

              + resulting state after opponent makes their move (None if game is over)

        **Return:**
            - None
        """

        s = tuple(s)

        if sp is not None:
            sp = tuple(sp)
            self.Q[s][a] += self.alpha * (reward + self.gamma*max(self.Q[sp].values()) - self.Q[s][a])
        else:
            self.Q[s][a] += self.alpha * (reward - self.Q[s][a])

    def train(self, games: int, save_games: int) -> None:
        """
        **Description:**
            - Make the Q agent play "games" games to train it. Continues building off of previous Q table.
              Thus, new accuracy measurements at each save point are concatenated to self.accuracy.

        **Parameters:**
            - games: int

              + games to play

            - save_games: int

              + save all data every save_games

        **Return:**
            - None
        """

        for game in range(1, games+1):
            # print(self.alpha)
            # self.alpha *= 0.9**(1/1000)

            if game % save_games == 0:
                # save updated object and update + save accuracy
                self.played_games += save_games

                self.accuracy.loc[self.accuracy.index[-1] + save_games] = self.test_accuracy() # new entry
                self.accuracy.to_csv(self.obj_dir + '/accuracy.csv')

                self.save(self.obj_dir + '/obj')

            game = Game(self.num_heaps, self.max_elements)
            turn = 0
            q_agent_last_action = None
            prev_heaps = None # heaps right before q agent's last action
            while not game.finished():
                turn += 1

                # find action that the current agent moving wants to do
                if turn & 1:
                    if random.uniform(0, 1) < self.epsilon:
                        # make random action
                        pos_actions = list(self.Q[tuple(game.heaps)].keys())
                        action = random.choice(pos_actions)
                    else:
                        # perform best move known to agent
                        action = self.action(game)

                    q_agent_last_action = action # save
                    prev_heaps = game.heaps.copy()

                    game.perform_action(*action)
                else:
                    action = self.train_opponent.action(game)
                    # perform action
                    game.perform_action(*action)

                    # update Q if game has not ended
                    if game.heaps and q_agent_last_action:
                        self.update_q(prev_heaps, q_agent_last_action, 0, game.heaps)

            # turn is now the total turns in the completed game
            win = turn & 1  # 1 if agent 1 won else 0 if agent 2 won

            # update Q now that game has ended
            if q_agent_last_action:
                reward = 1 if win else -1
                # lost, punish agent for making bad move and propagate expected future returns
                self.update_q(prev_heaps, q_agent_last_action, reward)

    def save(self, obj_file: str) -> None:
        """
        **Description:**
            - Pickles self.Q and saves it at obj_file
        """

        with open(obj_file, 'wb') as fout:
            pickle.dump(self, fout)

    def test_accuracy(self, num_actions: int = 1000) -> float:
        """
        **Description:**
            - Test num_actions on random game states to find percentage of optimal moves

        **Return:**
            - float representing the accuracy (optimal actions / num_actions)
        """

        accuracy = 0
        for action in range(num_actions):
            game = Game(self.num_heaps, self.max_elements)
            accuracy += game.perform_action(*self.action(game))
        accuracy /= num_actions

        return accuracy

    def plot_accuracy(self, color: str = None) -> None:
        """
        **Description:**
            - Plots self.accuracy.

        **Parameter:**
            - color: str

              + matplotlib color for plot
        """

        Ax = self.accuracy.plot()
        if color:
            Ax.get_lines()[0].set_color(color)
            Ax.get_legend().legendHandles[0].set_color(color)
        plt.xlabel(self.accuracy.index.name, color = "purple")
        plt.ylabel("Percentage of Optimal Moves", color = "purple")

    def save_accuracy_plot(self, color: str = None) -> None:
        """
        **Description:**
            - Plots self.accuracy and saves it.

        **Parameter:**
            - color: str

              + matplotlib color for plot
        """

        self.plot_accuracy(color)
        plt.savefig(self.obj_dir + f'/{self.played_games} games.png')

    def action(self, game: Game) -> tuple:
        """
        **Description:**
            - Given the current game, return the optimal action using the optimal policy

        **Parameter:**
            - game: Game

              + current game

        **Return:**
            - a tuple representing the action: (heap chosen, elements to remove)
        """

        heaps = tuple(game.heaps)
        best_action = None
        best_amt = -1e9
        for pos_action in self.Q[heaps]:
            curr_amt = self.Q[heaps][tuple(pos_action)]
            if curr_amt > best_amt:
                best_action = pos_action
                best_amt = curr_amt

        heap_chosen, elements_to_remove = best_action
        return heap_chosen, elements_to_remove

    def set_train_opponent(self, opponent_name: str) -> None:
        """
        **Description:**
            - Sets the self.train_opponent_name and self.train_opponent attributes

        **Parameter:**
            - opponent_name: str

              + Name of the opponent used to train the Q agent. Must be spelled exactly the same as the class name
                (make "self" for the agent to train by playing itself)
        """
        self.train_opponent_name = opponent_name

        if self.train_opponent_name == 'self':
            self.train_opponent = self
        else:
            exec('self.train_opponent = eval(self.train_opponent_name)()')
