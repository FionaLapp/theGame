# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 07:59:04 2021

@author: Fiona
"""

import cProfile
import re
import plotting
import pstats
from pstats import SortKey

number_of_cards_array=[*range(10, 50, 10)]
cProfile.run('plotting.TestStrategy.run_tests(number_of_cards_array)')
# cProfile.run('plotting.TestStrategy.run_tests(number_of_cards_array)', 'profiling')
# p = pstats.Stats('profiling')
# p.strip_dirs().sort_stats(SortKey.CUMULATIVE, SortKey.TIME).print_stats()