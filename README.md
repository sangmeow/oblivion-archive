# oblivion-archive

A curated repository of resources, datasets, and reference materials dedicated to enhancing GPT model training and fine-tuning.

## 작업근황

2025.08.09 - 소스 작동 확인 -> 입력 파일 준비 필요

## gpt-oss-20b에게 txt, md, pdf, docx, odf 파일로 학습 시키기

### gpt

```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/openai/gpt-oss-20b
**파일이 굉장히 큼**
```

### venv

```bash
python -m venv gpt-training
source gpt-training/bin/activate  # Linux/Mac
# gpt-training\Scripts\activate  # Windows
```

### 프로젝트 구조

```bash
mkdir oblivion-archive
cd oblivion-archive

# 폴더 구조 생성
mkdir documents
mkdir scripts
mkdir results
mkdir fine_tuned_model
```

### 0. 라이브러리 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install PyPDF2 python-docx odfpy
pip install peft  # LoRA를 위한 라이브러리
pip install wandb  # 학습 모니터링 (선택사항)
```

### 1. 문서 전처리

각 파일 형식을 텍스로 변화

```python
import os
import PyPDF2
import docx
from odf import text, teletype
from odf.opendocument import load

def extract_text_from_files(file_path):
    """다양한 파일 형식에서 텍스트 추출"""

    if file_path.endswith('.txt') or file_path.endswith('.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
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

    elif file_path.endswith('.odf') or file_path.endswith('.odt'):
        doc = load(file_path)
        paragraphs = doc.getElementsByType(text.P)
        return '\n'.join([teletype.extractText(p) for p in paragraphs])
```

### 2. 데이터셋 준비

학습할 데이터 가져오기

```python
from datasets import Dataset
import json

def prepare_training_data(data_dir):
    """학습 데이터 준비"""
    texts = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            try:
                text = extract_text_from_files(file_path)
                if text.strip():  # 빈 텍스트가 아닌 경우만
                    texts.append({"text": text})
            except Exception as e:
                print(f"파일 처리 중 오류: {filename} - {e}")

    return Dataset.from_list(texts)

# 데이터셋 생성
dataset = prepare_training_data("your_documents_folder")
```

여기 "your_documents_folder"에 학습할 데이터들을 담고 있는 폴더 디렉토리 경로를 입력한다.

### 3. 학습 시킬 LLM 토크나이저 및 모델 로드

토크나이저 : 사람이 읽는 텍스트를 컴퓨터가 이해할 수 있는 숫자로 변환. (예: "안녕하세요" → [1234, 5678, 9012] (숫자 토큰))

모델 로드: AI 두뇌(신경망) 메모리에 올리기

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# 모델과 토크나이저 로드
model_name = "your-gpt-oss-20b-path"  # 실제 모델 경로
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 패딩 토큰 설정 (필요한 경우)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 4. 데이터 전처리

```python
def tokenize_function(examples):
    """텍스트 토크나이징"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # 필요에 따라 조정
        return_tensors="pt"
    )

# 데이터셋 토크나이징
tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### 5. 학습 설정 및 실행

```python
from transformers import DataCollatorForLanguageModeling

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT는 causal LM이므로 False
)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # GPU 메모리에 따라 조정
    gradient_accumulation_steps=4,
    warmup_steps=500,
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # 메모리 절약
    dataloader_pin_memory=False,
)

# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```

### 통합

```python
#!/usr/bin/env python3
"""
GPT-OSS-20B 파인튜닝 스크립트
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
import json

# 라이브러리 import
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    from peft import get_peft_model, LoraConfig, TaskType
    import PyPDF2
    import docx
    from odf import text, teletype
    from odf.opendocument import load
except ImportError as e:
    print(f"필수 라이브러리가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치하세요:")
    print("pip install torch transformers datasets peft PyPDF2 python-docx odfpy")
    sys.exit(1)

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

class GPTTrainer:
    """GPT 모델 트레이너 클래스"""

    def __init__(self, model_name_or_path: str = "microsoft/DialoGPT-medium"):
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

    def tokenize_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
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
            gradient_accumulation_steps=8,
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

def main():
    """메인 함수"""
    try:
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            logger.info(f"GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            logger.info("CPU에서 실행됩니다.")

        # 문서 처리 및 데이터셋 준비
        documents_dir = "../documents"  # 문서가 있는 폴더 경로

        if not os.path.exists(documents_dir):
            raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {documents_dir}")

        dataset = DocumentProcessor.prepare_training_data(documents_dir)

        # 트레이너 초기화
        # 실제 GPT-OSS-20B 모델 경로가 있다면 여기에 입력하세요
        trainer = GPTTrainer("microsoft/DialoGPT-medium")

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
```
