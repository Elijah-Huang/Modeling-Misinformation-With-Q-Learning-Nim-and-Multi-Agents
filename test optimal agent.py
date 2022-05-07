def test_optimal_agent(num_games=10**4, max_num_heaps=100, max_elements=100) -> None:
    """Tests the optimal agent by playing 1e4 games with very large heaps (100 x 100).

    The optimal agent is tested against the random agent to ensure there is no logic overlap
    (nothing that we assume is true that we don't test).

    There is a high change the random agent code is correct and even a slight mistake
    should result in the optimal agent's loss.
    Thus, we can be confident there are no errors with the optimal agent if it passes.
    """

    for game_num in range(1, num_games + 1):
        game = Game(max_num_heaps, max_elements)
        win, full_game = game.play_game(OptimalAgent(), RandomAgent(), True)

        # although the xor sum of the intial heaps might not be 0,
        # all the optimal agent needs to win is for the random agent to play
        # a non-optimal move (not make xor sum = 0), which is almost guaranteed for the game size
        # the second agent should lose
        if win == 0:
            print(f"Bad Game:")
            for heaps in full_game:
                print(heaps)


if __name__ == "__main__":
    test_optimal_agent()  # does not print "Bad Game", so all games were won
