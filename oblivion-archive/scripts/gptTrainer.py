#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loggerSingleton import SingletonLogger as LoggerSingleton
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
import torch
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# 로깅 설정
singletonLogger = LoggerSingleton()
logger = singletonLogger.get_logger()


class GPTTrainer:
    """GPT 모델 트레이너 클래스"""
    
    def __init__(self, model_name_or_path: str = None ):
        """
        초기화
        실제 GPT-OSS-20B 모델 경로가 있다면 해당 경로를 사용하세요.
        예시로 DialoGPT를 사용합니다.
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.model = None
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        logger.info(f"모델 로딩 중: {self.model_name_or_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def setup_lora(self):
        """LoRA 설정으로 효율적인 파인튜닝"""
        logger.info("LoRA 설정 중...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]  # 모델에 따라 조정 필요
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA 설정 완료")
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 256) -> Dataset:
        """데이터셋 토크나이징"""
        logger.info("데이터셋 토크나이징 중...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"토크나이징 완료: {len(tokenized_dataset)}개 샘플")
        return tokenized_dataset
    
    def train(self, dataset: Dataset, output_dir: str = "./results"):
        """모델 학습"""
        logger.info("학습 시작...")
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 학습 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # GPU 메모리에 따라 조정
            gradient_accumulation_steps=16,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            report_to=None,  # wandb 사용하려면 "wandb"로 변경
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # 학습 실행
        trainer.train()
        
        # 모델 저장
        output_model_dir = "./fine_tuned_model"
        trainer.save_model(output_model_dir)
        self.tokenizer.save_pretrained(output_model_dir)
        
        logger.info(f"학습 완료! 모델이 {output_model_dir}에 저장되었습니다.")
