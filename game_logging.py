# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 11:46:15 2021

@author: Fiona
"""
import logging
import sys

# %% configuration


class LoggingMethods():
    def createLogger(name, level, file=None, console_logging=False):
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if file is not None:
            # create console handler and set level to debug
            handler = logging.FileHandler(file)
            handler.setLevel(level)
            # create formatter
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            # add formatter to handler
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            # console_handler.set_level(level)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger


class ContextManager():
    def __init__(self):
        # GameLoggers.debug_logger.handlers.clear()
        # os.remove('debug.log')
        # GameLoggers.strategy_logger.handlers.clear()
        # os.remove('strategy.log')
        GameLoggers.debug_logger.debug('Context Manager init method called')

    def __enter__(self):
        GameLoggers.debug_logger.debug('enter method called')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        GameLoggers.debug_logger.debug('closing loggers')
        GameLoggers.debug_logger.handlers.clear()
        GameLoggers.strategy_logger.handlers.clear()


class GameLoggers():
    # change log_level to debug in debug_logger to create output in debug.log
    debug_logger = LoggingMethods.createLogger(
        "debug_logger", logging.INFO, file="debug.log", console_logging=False)
    strategy_logger = LoggingMethods.createLogger(
        "strategy_logger", logging.INFO, file="strategy.log",
        console_logging=False)


