# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 07:59:04 2021

@author: Fiona
"""

import profile
import cProfile
import re
import plotting
import pstats
from pstats import SortKey
import game_logging
import game_test



number_of_cards_array=[*range(10, 110, 10)]
cProfile.run('plotting.TestStrategy.run_tests(number_of_cards_array)', 'profiling_log')
#cProfile.run('logging_test()', 'profiling')

p = pstats.Stats('profiling_log')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE, SortKey.TIME).print_stats()

