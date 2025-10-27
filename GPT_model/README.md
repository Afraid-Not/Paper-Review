# GPT-1 기반 한국어-영어 번역 모델

이 프로젝트는 GPT-1 아키텍처를 사용하여 한국어를 영어로 번역하는 모델을 구현합니다.

## 목표
"나는 밥을 먹는다" → "I have breakfast" 번역

## 설치 및 실행

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 실행
```bash
python gpt.py
```

## 모델 구조

- **GPT-1 아키텍처**: Transformer Decoder 기반
- **모델 크기**: 512 차원, 8개 어텐션 헤드, 6개 레이어
- **어휘 크기**: 10,000개 토큰
- **최대 시퀀스 길이**: 128

## 주요 기능

1. **간단한 토크나이저**: 한국어/영어 텍스트를 토큰으로 변환
2. **위치 인코딩**: 시퀀스 내 위치 정보 추가
3. **Causal Mask**: GPT 스타일의 자기회귀적 생성
4. **Teacher Forcing**: 훈련 시 정답 시퀀스 사용

## 훈련 데이터

15개의 한국어-영어 번역 쌍을 포함:
- "나는 밥을 먹는다" → "I have breakfast"
- "나는 물을 마신다" → "I drink water"
- "나는 책을 읽는다" → "I read a book"
- 등등...

## 사용법

모델 훈련 후 자동으로 번역 테스트가 실행됩니다:

```python
# 직접 번역 함수 사용
translation = translate(model, "나는 밥을 먹는다", korean_tokenizer, english_tokenizer)
print(translation)  # "I have breakfast"
```

## 모델 저장

훈련된 모델은 `gpt_translation_model.pth`로 저장되며, 다음을 포함합니다:
- 모델 가중치
- 한국어/영어 토크나이저
- 설정 정보

## 하이퍼파라미터

- 학습률: 0.0001
- 배치 크기: 32
- 에포크: 100
- 드롭아웃: 0.1
