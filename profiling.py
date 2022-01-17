# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 07:59:04 2021

@author: Fiona

This file can be used to benchmark and tweak game-performance.
The aim is to run the game a set number of times, and use CProfile to find out
how much time each function takes, cumulatively, and how often it is called.
Running the code will print a table to screen.
Note that the function-name can be found in the last column
(filename:lineno(function))
cumtime refers to the cumulative time spent in each function - note that this
includes time spent in subfunctions. Try to identify subfunctions that are far
up the list and figure out if we can optimise those (rather than tweaking
functions that don't have a large impact in the first case)
ncalls refers to the number of times the function has been called in total
(e.g. for trying to minimise the number of times we check if the game is
finished)
We order the table by cumtime (uncomment below to order by ncalls instead)
For this file, we test by runing the run_tests() method of the plotting
file - this runs 4*100 tests per element
(points_of_error_calculation* plotting.NUMBER_OF_ATTEMPTS).
We run this on the number_of_cards_array vector, which has game sizes
from 10 to 100 cards (in 10-card increments), so the game is played a
total of 400*10 times (but the first few will take less time since there
are less cards)


"""

import cProfile
import pstats
import plotting
import the_game
import test_deep_q_rl
from pstats import SortKey

if __name__ == "__main__":
    # initialise a vector with number of cards [10,20,30,40,50,60,70,80,90,100]
    number_of_cards_array = [*range(10, 110, 10)]
    strategies = [the_game.PlayWithMetricStrategy]

    # cProfile.run(
    #     ('plotting.TestStrategy.run_tests(strategies, number_of_cards_array, '
    #      'variable="number_of_cards", points_for_error_calculation=4)'),
    #     'profiling_log')

    cProfile.run('test_deep_q_rl.run_tests()','profiling_log')

    p = pstats.Stats('profiling_log')
    # sort by cumulative time
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE, SortKey.TIME).reverse_order().print_stats()

    # # sort by number of calls performed by each function
    # p.strip_dirs().sort_stats(SortKey.CALLS).print_stats()
