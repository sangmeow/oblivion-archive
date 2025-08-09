#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_model():
    """학습된 모델 테스트"""
    
    # 베이스 모델 로드
    base_model_name = "microsoft/DialoGPT-medium"  # 원래 사용한 베이스 모델
    model_path = ".//fine_tuned_model"  # 파인튜닝된 모델 경로
    
    print("모델 로딩 중...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA 모델 로드
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("모델 로딩 완료!")
    
    # 테스트 프롬프트
    test_prompts = [
        "안녕하세요",
        "AI에 대해 설명해주세요",
        "파이썬 프로그래밍이란"
    ]
    
    for prompt in test_prompts:
        print(f"\n프롬프트: {prompt}")
        print("-" * 50)
        
        # 입력 토크나이징
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        print(f"응답: {response}")

if __name__ == "__main__":
    test_model()