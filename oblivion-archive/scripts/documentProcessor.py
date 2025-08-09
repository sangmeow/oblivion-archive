#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import PyPDF2
import docx
from json import load
from odf import teletype
from datasets import Dataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """문서 처리 클래스"""
    
    @staticmethod
    def extract_text_from_files(file_path: str) -> str:
        """다양한 파일 형식에서 텍스트 추출"""
        try:
            if file_path.endswith(('.txt', '.md')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            elif file_path.endswith(('.odf', '.odt')):
                doc = load(file_path)
                paragraphs = doc.getElementsByType(text.P)
                return '\n'.join([teletype.extractText(p) for p in paragraphs])
            
            else:
                logger.warning(f"지원하지 않는 파일 형식: {file_path}")
                return ""
                
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생 {file_path}: {e}")
            return ""

    @staticmethod
    def prepare_training_data(data_dir: str) -> Dataset:
        """학습 데이터 준비"""
        texts = []
        processed_files = 0
        
        logger.info(f"문서 폴더에서 데이터 로딩 중: {data_dir}")
        
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                text = DocumentProcessor.extract_text_from_files(file_path)
                if text.strip():  # 빈 텍스트가 아닌 경우만
                    texts.append({"text": text})
                    processed_files += 1
                    logger.info(f"처리 완료: {filename} (길이: {len(text)} 문자)")
        
        logger.info(f"총 {processed_files}개 파일 처리 완료")
        
        if not texts:
            raise ValueError("처리할 수 있는 문서가 없습니다. documents 폴더를 확인하세요.")
        
        return Dataset.from_list(texts)
