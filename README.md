# Paper Review Repository

> Deep Learning 연구 논문 구현 및 재현 아카이브
> 김재현 (Jaehyeon Kim) | AI Research & Implementation Study

---

## Repository Overview

이 저장소는 딥러닝 주요 논문을 직접 읽고 PyTorch 기반으로 **구현·재현·시각화**한 결과를 정리한 개인 스터디 리포지토리입니다.

생성 모델(GAN → VAE → Diffusion), 객체 탐지(YOLO, RT-DETR, Two-Stage), 언어 모델(GPT, BERT, Transformer),
추론 에이전트(ReasoningBank) 등 다양한 분야를 다루며,
COCO128 / CIFAR10 / UTKFace / HotpotQA 등의 공개 데이터셋을 활용해 실험을 진행했습니다.

---

## Directory Structure

```
Paper-Review/
├── Diffusion/          # GAN, VAE, DDPM, DDIM 구현
├── YOLO/               # YOLOv1/v3/v5/v11 + 얼굴 검출 파이프라인
├── GPT/                # GPT-1/2/3, BERT 구현 (한영 번역)
├── Transformer/        # Attention is All You Need 완전 구현
├── ReasoningBank/      # ReAct + Reflexion + ReasoningBank 에이전트
├── SAM2/               # Segment Anything v2 테스트
├── CV/                 # Vision preprocessing 논문 리뷰 (PDF)
├── Power_Retention/    # Power Retention 논문 리뷰 (PDF)
├── RT-DETR/            # RT-DETR 논문 리뷰 (PDF)
└── TwoStage/           # Two-Stage Detector 논문 리뷰 (PDF)
```

---

## Implemented Papers

| 분야                  | 모델                              | 논문명 / 구현 내용                                               |
| --------------------- | --------------------------------- | ---------------------------------------------------------------- |
| **Generative Models** | **GAN**                           | _Generative Adversarial Networks_ (Goodfellow et al., 2014)      |
|                       | **VAE**                           | _Auto-Encoding Variational Bayes_ (Kingma & Welling, 2013)       |
|                       | **DDPM / DDIM**                   | _Denoising Diffusion Probabilistic Models / Implicit Models_     |
| **Object Detection**  | **YOLOv1–v11**                    | _You Only Look Once_ (Redmon et al., 2016) → Ultralytics YOLOv11 |
|                       | **RT-DETR**                       | _Real-Time Detection Transformer_ (논문 리뷰)                    |
|                       | **Two-Stage**                     | _Faster R-CNN_ 등 Two-Stage Detector (논문 리뷰)                 |
| **Segmentation**      | **SAM2**                          | _Segment Anything Model v2_ (Meta AI, 2024)                      |
| **Language Models**   | **GPT-1/2/3**                     | _Improving Language Understanding by Generative Pre-Training_ 등 |
|                       | **BERT**                          | _Pre-training of Deep Bidirectional Transformers_                |
|                       | **Transformer**                   | _Attention Is All You Need_ (Vaswani et al., 2017)               |
| **Reasoning**         | **ReasoningBank**                 | ReAct + Reflexion + ReasoningBank 통합 에이전트                  |
| **Face Recognition**  | **MTCNN + FaceNet + YOLOv11-CLS** | 얼굴 검출 → 임베딩 → 자동 라벨링 → YOLO 분류 파인튜닝            |
| **Vision**            | **CV Preprocessing**              | Vision preprocessing 논문 리뷰                                   |
| **Retention**         | **Power Retention**               | Power Retention 논문 리뷰                                        |

---

## Key Features

- **End-to-End 구현**: 논문 구조를 PyTorch 코드로 완전 재현
- **모듈 단위 비교**: `GAN.py`, `VAE.py`, `DDPM.py` 등은 동일한 데이터로 결과 비교 가능
- **시각화 중심 학습**: 모든 실험 결과를 이미지(`*_results.png`)로 저장
- **자동화된 실험 파이프라인**: COCO128 자동 다운로드 + 결과 폴더 자동 정리
- **한글 친화적 시각화**: Matplotlib + PIL 폰트 설정으로 그래프/박스에 한글 지원

---

## Environment

```bash
# Base Environment
conda create -n paperreview python=3.10
conda activate paperreview

# Core Dependencies
uv pip install torch torchvision torchaudio
uv pip install opencv-python matplotlib numpy tqdm pyyaml
uv pip install ultralytics facenet-pytorch albumentations

# NLP / LLM (GPT, Transformer, ReasoningBank)
uv pip install transformers accelerate datasets huggingface-hub openai pandas
```

---

## Author

**김재현 (Jaehyeon Kim)**

> AI Researcher / Developer
> Focus: Deep Learning, Computer Vision, Generative Models, Representation Learning
> Seoul, Republic of Korea
> GitHub: [PlusMinusAnd](https://github.com/PlusMinusAnd)

---

## License

This repository is for **academic and educational purposes** only.
All code and assets are © 2025 Jaehyeon Kim unless otherwise stated.

---

> "읽고 → 이해하고 → 재현하고 → 설명한다."
> — 이것이 저의 연구 루틴입니다.
