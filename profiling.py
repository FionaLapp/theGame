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



number_of_cards_array=[*range(10, 110, 10)]
cProfile.run('plotting.TestStrategy.run_tests(number_of_cards_array)', 'profiling_log')

p = pstats.Stats('profiling_log')
# sort by cumulative time
p.strip_dirs().sort_stats(SortKey.CUMULATIVE, SortKey.TIME).print_stats()

# # sort by number of calls performed by each function
# p.strip_dirs().sort_stats(SortKey.CALLS).print_stats()

