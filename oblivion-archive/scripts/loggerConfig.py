# -*- coding: utf-8 -*-

import logging
import sys

class LoggerConfig:
    """로깅 설정을 관리하는 클래스 (한글 지원)"""
    
    def __init__(self, log_file='training.log', level=logging.INFO, encoding='utf-8'):
        """
        LoggerConfig 초기화
        
        Args:
            log_file (str): 로그 파일명 (기본값: 'training.log')
            level: 로깅 레벨 (기본값: logging.INFO)
            encoding (str): 파일 인코딩 (기본값: 'utf-8')
        """
        self.log_file = log_file
        self.level = level
        self.encoding = encoding
        self.logger = None
        
    def setup_logging(self):
        """로깅 설정을 구성하고 logger를 반환 (한글 지원)"""
        # 기존 핸들러들 제거 (중복 방지)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 파일 핸들러 (UTF-8 인코딩으로 한글 지원)
        file_handler = logging.FileHandler(
            self.log_file, 
            encoding=self.encoding
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # 콘솔 핸들러 (한글 출력 지원)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # 루트 로거 설정
        logging.root.setLevel(self.level)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        
        # logger 인스턴스 생성
        self.logger = logging.getLogger(__name__)
        return self.logger
    
    def get_logger(self, name=None):
        """특정 이름으로 logger를 가져오기"""
        if name:
            return logging.getLogger(name)
        return self.logger if self.logger else self.setup_logging()
    
    def change_log_level(self, level):
        """로그 레벨 변경"""
        self.level = level
        if self.logger:
            self.logger.setLevel(level)
    
    def add_file_handler(self, filename, encoding='utf-8'):
        """새로운 파일 핸들러 추가 (한글 지원)"""
        if self.logger:
            file_handler = logging.FileHandler(filename, encoding=encoding)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def set_korean_formatter(self):
        """한글 날짜 형식을 포함한 포매터 설정"""
        korean_formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 모든 핸들러에 한글 지원 포매터 적용
        for handler in logging.root.handlers:
            handler.setFormatter(korean_formatter)