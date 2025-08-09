#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from loggerSingleton import SingletonLogger as LoggerSingleton

import torch
from documentProcessor import DocumentProcessor
from gptTrainer import GPTTrainer

# 로깅 설정
singletonLogger = LoggerSingleton()
logger = singletonLogger.get_logger()

# Constants
TRAINING_LOG_FILE = 'training.log'
DOCUMENTS_DIR = "../documents"  # 문서가 있는 폴더 경로
GPTTrainer_DIR = "../../models/gpt-oss-20b"  # GPT 모델 경로

def main():
    """메인 함수"""
    try:
        
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            logger.info(f"GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            logger.info("CPU에서 실행됩니다.")
        
        # 문서 처리 및 데이터셋 준비
        if not os.path.exists(DOCUMENTS_DIR):
            raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {DOCUMENTS_DIR}")
        
        dataset = DocumentProcessor.prepare_training_data(DOCUMENTS_DIR)
        
        # 트레이너 초기화
        # 실제 GPT-OSS-20B 모델 경로가 있다면 여기에 입력하세요
        trainer = GPTTrainer(GPTTrainer_DIR)
        
        # 모델 로드 및 설정
        trainer.load_model_and_tokenizer()
        trainer.setup_lora()  # 메모리 절약을 위한 LoRA 사용
        
        # 데이터 토크나이징
        tokenized_dataset = trainer.tokenize_dataset(dataset)
        
        # 학습 실행
        trainer.train(tokenized_dataset)
        
        logger.info("모든 과정이 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()