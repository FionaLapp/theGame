# theGame

Mom got us [this card game](https://de.wikipedia.org/wiki/The_Game_(Kartenspiel)) for Christmas and Clara mentioned that she'd tried (and given up on) coding a strategy for it, so obviously the only logical response was to spend Christmas break obsessing about it.

## Aims
- Write code that can simulate the game (and do so as efficiently as possible)
- Write a simple strategy to play the game, based off of what a human would do (but less complicated).
- Do a little bit of plotting and see how different variables impact the odds of winning 
- If time allows (which it does, of course ... the question is if motivation also allows), let a reinforcement learning algorithm loose on the game and figure out how well it can play

## Contents
- game.py a file containing game logic, as well as some basic hard-coded strategies
- game_test.py a few unit tests to be run after any changes in game.py. To run, use "pytest game_test.py" in the directory where the file is located (needs more tests)
- game_logging.py change values in this file to modify logging behaviour (needs to be cleaned up /commented slightly)
- plotting.py some initial plots to get an understanding of the impact of parameters (needs to be cleaned up)
- profiling.py running this will help identify where calculations can save time (needs to be cleaned up)
- reinforcement_learning.py (currently not working yet)
