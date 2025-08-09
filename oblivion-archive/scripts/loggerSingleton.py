# -*- coding: utf-8 -*-

from loggerConfig import LoggerConfig

class SingletonLogger:
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            log_config = LoggerConfig(encoding='utf-8')
            cls._logger = log_config.setup_logging()
        return cls._instance
    
    def get_logger(self):
        return self._logger