# Paper-Review

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-0099FF?logo=yolo&logoColor=white)](https://ultralytics.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://openai.com/)
[![License](https://img.shields.io/badge/License-Educational-green)](#%EB%9D%BC%EC%9D%B4%EC%84%A0%EC%8A%A4)

> **AI 논문 구현 아카이브** -- Deep Learning 주요 논문을 직접 읽고, PyTorch로 구현/재현/시각화한 개인 스터디 리포지토리

**김재현 (Jaehyeon Kim)** | GitHub: [Afraid-Not](https://github.com/Afraid-Not)

---

## 목차

- [프로젝트 소개](#프로젝트-소개)
- [모듈 개요](#모듈-개요)
- [코드 구현 모듈](#코드-구현-모듈)
  - [Diffusion](#diffusion)
  - [YOLO](#yolo)
  - [GPT](#gpt)
  - [Transformer](#transformer)
  - [ReasoningBank](#reasoningbank)
  - [SAM2](#sam2)
- [PDF 리뷰 모듈](#pdf-리뷰-모듈)
- [기술 스택](#기술-스택)
- [실행 방법](#실행-방법)
- [프로젝트 구조](#프로젝트-구조)
- [코드 컨벤션](#코드-컨벤션)
- [라이선스](#라이선스)

---

## 프로젝트 소개

이 저장소는 딥러닝 핵심 논문들을 **직접 읽고 PyTorch 기반으로 구현/재현/시각화**한 결과를 정리한 아카이브입니다.

다루는 분야:

- **생성 모델** -- GAN, VAE, DDPM, DDIM (Diffusion Model 계열 진화 과정)
- **객체 탐지** -- YOLOv1/v3/v5/v11, RT-DETR, Two-Stage Detector
- **언어 모델** -- GPT-1/2/3, BERT, Transformer
- **추론 에이전트** -- ReAct + Reflexion + ReasoningBank (HotpotQA)
- **세그멘테이션** -- Segment Anything v2 (SAM2)
- **얼굴 검출** -- MTCNN + FaceNet 기반 파이프라인
- **비전 전처리** -- CV Preprocessing, Power Retention

활용 데이터셋: COCO128, CIFAR-10, UTKFace, HotpotQA 등 공개 데이터셋 (스크립트 실행 시 자동 다운로드)

---

## 모듈 개요

| 모듈                              | 유형       | 핵심 내용                                   | 논문 리뷰 PDF                                                                             | 구현 PDF                                                    |
| --------------------------------- | ---------- | ------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Diffusion](#diffusion)           | Code + PDF | GAN, VAE, DDPM, DDIM -- 생성 모델 진화 과정 | [Paper Review](./Diffusion/논문_삼석사와아이들_김재현_DiffusionModelPaperReview_1014.pdf) | [구현 문서](./Diffusion/논문구현_Diffusion.pdf)             |
| [YOLO](#yolo)                     | Code + PDF | YOLOv1/v3/v5/v11 + MTCNN 얼굴 검출          | [Paper Review](./YOLO/논문_삼석사와아이들_김재현_YoloPaperReview_1001.pdf)                | [구현 문서](./YOLO/논문구현_Yolo.pdf)                       |
| [GPT](#gpt)                       | Code + PDF | GPT-1/2/3 + BERT -- 한영 번역               | [Paper Review](./GPT/GPT_BERT_Paper_Review.pdf)                                           | [구현 문서](./GPT/논문구현_GPTBERT.pdf)                     |
| [Transformer](#transformer)       | Code + PDF | Attention is All You Need 완전 구현         | [Paper Review](./Transformer/Transformer_Paper_review.pdf)                                | [구현 문서](./Transformer/논문구현_Transformer.pdf)         |
| [ReasoningBank](#reasoningbank)   | Code + PDF | ReAct + Reflexion 에이전트, HotpotQA 통합   | --                                                                                        | [구현 문서](./ReasoningBank/논문구현_ReasoningBank.pdf)     |
| [SAM2](#sam2)                     | Code + PDF | Segment Anything v2 -- 그리드 + 인터랙티브  | --                                                                                        | [구현 문서](./SAM2/논문구현_SAM2.pdf)                       |
| [CV](#pdf-리뷰-모듈)              | PDF only   | Vision Preprocessing 논문 리뷰              | --                                                                                        | [구현 문서](./CV/논문구현_vision_preprocessing.pdf)         |
| [Power_Retention](#pdf-리뷰-모듈) | PDF only   | Power Retention 논문 리뷰                   | --                                                                                        | [구현 문서](./Power_Retention/논문구현_Power_Retention.pdf) |
| [RT-DETR](#pdf-리뷰-모듈)         | PDF only   | Real-Time Detection Transformer 리뷰        | --                                                                                        | [구현 문서](./RT-DETR/논문구현_RT-DETR.pdf)                 |
| [TwoStage](#pdf-리뷰-모듈)        | PDF only   | Two-Stage Detector 리뷰                     | --                                                                                        | [구현 문서](./TwoStage/논문구현_Two-stage.pdf)              |

---

## 코드 구현 모듈

### Diffusion

> GAN -> VAE -> DDPM -> DDIM 으로 이어지는 생성 모델 계열의 진화 과정을 추적 구현

**PDF 문서:**

- [논문 리뷰 -- Diffusion Model Paper Review](./Diffusion/논문_삼석사와아이들_김재현_DiffusionModelPaperReview_1014.pdf)
- [논문 구현 문서](./Diffusion/논문구현_Diffusion.pdf)

**구현 파일:**

| 파일                                 | 설명                                                      |
| ------------------------------------ | --------------------------------------------------------- |
| [`GAN.py`](./Diffusion/GAN.py)       | Generative Adversarial Networks (Goodfellow et al., 2014) |
| [`VAE.py`](./Diffusion/VAE.py)       | Auto-Encoding Variational Bayes (Kingma & Welling, 2013)  |
| [`DDPM.py`](./Diffusion/DDPM.py)     | Denoising Diffusion Probabilistic Models                  |
| [`DDIM.py`](./Diffusion/DDIM.py)     | Denoising Diffusion Implicit Models (InstructPix2Pix)     |
| [`fixing.py`](./Diffusion/fixing.py) | 보조 유틸리티                                             |

---

### YOLO

> YOLOv1부터 v11까지의 객체 탐지 모델 구현 + MTCNN 기반 얼굴 검출 파이프라인

**PDF 문서:**

- [논문 리뷰 -- YOLO Paper Review](./YOLO/논문_삼석사와아이들_김재현_YoloPaperReview_1001.pdf)
- [논문 구현 문서](./YOLO/논문구현_Yolo.pdf)

**구현 파일:**

| 파일                                                | 설명                            |
| --------------------------------------------------- | ------------------------------- |
| [`YOLOv1.py`](./YOLO/YOLOv1.py)                     | You Only Look Once v1 (COCO128) |
| [`YOLOv3.py`](./YOLO/YOLOv3.py)                     | YOLOv3 구현                     |
| [`YOLOv5.py`](./YOLO/YOLOv5.py)                     | YOLOv5 (mAP 평가 포함)          |
| [`YOLOv11.py`](./YOLO/YOLOv11.py)                   | Ultralytics YOLOv11             |
| [`face_detecting.py`](./YOLO/face_detecting.py)     | MTCNN 기반 얼굴 검출            |
| [`face_crops.py`](./YOLO/face_crops.py)             | 얼굴 크롭 처리                  |
| [`face_detecting_2.py`](./YOLO/face_detecting_2.py) | 얼굴 검출 확장 v2               |
| [`face_detecting_3.py`](./YOLO/face_detecting_3.py) | 얼굴 검출 확장 v3               |

---

### GPT

> GPT-1/2/3 및 BERT를 PyTorch로 구현하고 한영 번역 태스크에 적용

**PDF 문서:**

- [논문 리뷰 -- GPT & BERT Paper Review](./GPT/GPT_BERT_Paper_Review.pdf)
- [논문 구현 문서](./GPT/논문구현_GPTBERT.pdf)

**구현 파일:**

| 파일                       | 설명                                                               |
| -------------------------- | ------------------------------------------------------------------ |
| [`gpt.py`](./GPT/gpt.py)   | GPT-1: Improving Language Understanding by Generative Pre-Training |
| [`gpt2.py`](./GPT/gpt2.py) | GPT-2: Language Models are Unsupervised Multitask Learners         |
| [`gpt3.py`](./GPT/gpt3.py) | GPT-3: Language Models are Few-Shot Learners                       |
| [`bert.py`](./GPT/bert.py) | BERT: Pre-training of Deep Bidirectional Transformers              |

---

### Transformer

> "Attention Is All You Need" (Vaswani et al., 2017) 논문의 완전한 PyTorch 구현

**PDF 문서:**

- [논문 리뷰 -- Transformer Paper Review](./Transformer/Transformer_Paper_review.pdf)
- [논문 구현 문서](./Transformer/논문구현_Transformer.pdf)

**구현 파일:**

| 파일                                 | 설명                                                                 |
| ------------------------------------ | -------------------------------------------------------------------- |
| [`trans.py`](./Transformer/trans.py) | Multi-Head Attention, Positional Encoding, Encoder-Decoder 전체 구현 |

---

### ReasoningBank

> ReAct + Reflexion + ReasoningBank 추론 에이전트를 HotpotQA에 적용하고, 조합별 성능 비교 분석

**PDF 문서:**

- [논문 구현 문서](./ReasoningBank/논문구현_ReasoningBank.pdf)

**구현 파일:**

| 파일                                                                                       | 설명                                            |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| [`react_reflexion_reasoningbank.py`](./ReasoningBank/react_reflexion_reasoningbank.py)     | ReAct + Reflexion + ReasoningBank 통합 에이전트 |
| [`hotpot_qa_with_llm.py`](./ReasoningBank/hotpot_qa_with_llm.py)                           | OpenAI API 기반 HotpotQA 풀이                   |
| [`hotpot_qa_integration.py`](./ReasoningBank/hotpot_qa_integration.py)                     | HotpotQA 통합 파이프라인                        |
| [`compare_combinations.py`](./ReasoningBank/compare_combinations.py)                       | 에이전트 조합 비교 실험                         |
| [`compare_all_combinations_hotpot.py`](./ReasoningBank/compare_all_combinations_hotpot.py) | 전체 조합 비교 (HotpotQA)                       |
| [`example_usage.py`](./ReasoningBank/example_usage.py)                                     | 사용 예시                                       |
| [`hotpot.py`](./ReasoningBank/hotpot.py)                                                   | HotpotQA 데이터 로더                            |

**의존성 파일:**

| 파일                                                                 | 설명              |
| -------------------------------------------------------------------- | ----------------- |
| [`hotpot_requirements.txt`](./ReasoningBank/hotpot_requirements.txt) | HotpotQA 실험용   |
| [`openai_requirements.txt`](./ReasoningBank/openai_requirements.txt) | OpenAI API 연동용 |
| [`llama_requirements.txt`](./ReasoningBank/llama_requirements.txt)   | LLaMA 모델 연동용 |

**분석 결과:**

- [comparison_table.md](./ReasoningBank/comparison_table.md) -- 에이전트 조합 성능 비교표
- [hotpot_all_combinations_table.md](./ReasoningBank/hotpot_all_combinations_table.md) -- HotpotQA 전체 조합 비교표

---

### SAM2

> Segment Anything Model v2 (Meta AI, 2024) -- 그리드 기반 자동 + 인터랙티브 세그멘테이션 테스트

**PDF 문서:**

- [논문 구현 문서](./SAM2/논문구현_SAM2.pdf)

**구현 파일:**

| 파일                                  | 설명                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------ |
| [`sam2_test.py`](./SAM2/sam2_test.py) | 그리드 기반 자동 세그멘테이션 + 포인트/박스 기반 인터랙티브 세그멘테이션 |

---

## PDF 리뷰 모듈

코드 구현 없이 논문 리뷰 문서만 포함된 모듈입니다.

| 모듈                | 논문 주제                                 | PDF 링크                                                                        |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| **CV**              | Vision Preprocessing 기법 리뷰            | [논문구현\_vision_preprocessing.pdf](./CV/논문구현_vision_preprocessing.pdf)    |
| **Power_Retention** | Power Retention 메커니즘 리뷰             | [논문구현\_Power_Retention.pdf](./Power_Retention/논문구현_Power_Retention.pdf) |
| **RT-DETR**         | Real-Time Detection Transformer 리뷰      | [논문구현\_RT-DETR.pdf](./RT-DETR/논문구현_RT-DETR.pdf)                         |
| **TwoStage**        | Two-Stage Detector (Faster R-CNN 등) 리뷰 | [논문구현\_Two-stage.pdf](./TwoStage/논문구현_Two-stage.pdf)                    |

---

## 기술 스택

| 분류                 | 기술                                           |
| -------------------- | ---------------------------------------------- |
| **Deep Learning**    | PyTorch, torchvision, torchaudio               |
| **Object Detection** | Ultralytics (YOLO), facenet-pytorch (MTCNN)    |
| **NLP / LLM**        | HuggingFace Transformers, accelerate, datasets |
| **API**              | OpenAI API                                     |
| **Computer Vision**  | OpenCV, PIL                                    |
| **Visualization**    | matplotlib                                     |
| **Utilities**        | numpy, tqdm, pandas, PyYAML, albumentations    |

---

## 실행 방법

모든 스크립트는 독립 실행 파일입니다. CLI 인자 없이 직접 실행할 수 있습니다.

```bash
# 환경 생성
conda create -n paperreview python=3.10
conda activate paperreview

# 핵심 패키지 설치
uv pip install torch torchvision torchaudio
uv pip install opencv-python matplotlib numpy tqdm pyyaml Pillow
uv pip install ultralytics facenet-pytorch albumentations

# NLP / LLM 패키지 (GPT, Transformer, ReasoningBank)
uv pip install transformers accelerate datasets huggingface-hub openai pandas

# 스크립트 실행
python Diffusion/DDPM.py
python YOLO/YOLOv1.py
python GPT/gpt.py
python Transformer/trans.py
```

- CUDA/CPU 자동 감지
- 데이터셋 자동 다운로드 (COCO128, CIFAR-10, UTKFace, HotpotQA)

---

## 프로젝트 구조

```
Paper-Review/
├── Diffusion/              # GAN, VAE, DDPM, DDIM 구현
│   ├── GAN.py
│   ├── VAE.py
│   ├── DDPM.py
│   ├── DDIM.py
│   ├── fixing.py
│   └── *.pdf
│
├── YOLO/                   # YOLOv1/v3/v5/v11 + 얼굴 검출
│   ├── YOLOv1.py
│   ├── YOLOv3.py
│   ├── YOLOv5.py
│   ├── YOLOv11.py
│   ├── face_detecting.py
│   ├── face_crops.py
│   ├── face_detecting_2.py
│   ├── face_detecting_3.py
│   └── *.pdf
│
├── GPT/                    # GPT-1/2/3 + BERT
│   ├── gpt.py
│   ├── gpt2.py
│   ├── gpt3.py
│   ├── bert.py
│   ├── requirements.txt
│   └── *.pdf
│
├── Transformer/            # Attention is All You Need
│   ├── trans.py
│   └── *.pdf
│
├── ReasoningBank/          # ReAct + Reflexion 에이전트
│   ├── react_reflexion_reasoningbank.py
│   ├── hotpot_qa_with_llm.py
│   ├── hotpot_qa_integration.py
│   ├── compare_combinations.py
│   ├── compare_all_combinations_hotpot.py
│   ├── example_usage.py
│   ├── hotpot.py
│   ├── *_requirements.txt
│   └── *.pdf
│
├── SAM2/                   # Segment Anything v2
│   ├── sam2_test.py
│   └── *.pdf
│
├── CV/                     # Vision Preprocessing (PDF only)
├── Power_Retention/        # Power Retention (PDF only)
├── RT-DETR/                # RT-DETR (PDF only)
└── TwoStage/               # Two-Stage Detector (PDF only)
```

---

## 코드 컨벤션

- 각 디렉토리는 **독립적** -- 디렉토리 간 import 없음
- 스크립트 구조: Imports -> Config -> Dataset -> Model -> Training -> Inference -> Visualization -> `__main__`
- 모든 문서와 주석은 **한국어**로 작성
- 시각화는 한글 폰트 설정 적용 (Matplotlib + PIL)
- ReasoningBank는 에피소드별 **구조화된 JSON 로그** 출력

---

## 라이선스

이 저장소는 **학습 및 교육 목적**으로만 사용 가능합니다.
모든 코드와 문서의 저작권은 2025 김재현(Jaehyeon Kim)에게 있습니다.

---

---

# Paper-Review

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-0099FF?logo=yolo&logoColor=white)](https://ultralytics.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://openai.com/)
[![License](https://img.shields.io/badge/License-Educational-green)](#license)

> **Deep Learning Paper Implementation Archive** -- A personal study repository of paper-to-code implementations in PyTorch

**Jaehyeon Kim** | GitHub: [Afraid-Not](https://github.com/Afraid-Not)

---

## Table of Contents

- [About](#about)
- [Module Overview](#module-overview)
- [Code Implementation Modules](#code-implementation-modules)
  - [Diffusion](#diffusion-1)
  - [YOLO](#yolo-1)
  - [GPT](#gpt-1)
  - [Transformer](#transformer-1)
  - [ReasoningBank](#reasoningbank-1)
  - [SAM2](#sam2-1)
- [PDF-Only Review Modules](#pdf-only-review-modules)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Code Conventions](#code-conventions)
- [License](#license)

---

## About

This repository is an archive of **standalone PyTorch implementations** of key deep learning papers, accompanied by Korean-language review documents.

Covered areas:

- **Generative Models** -- GAN, VAE, DDPM, DDIM (tracing the evolution of diffusion models)
- **Object Detection** -- YOLOv1/v3/v5/v11, RT-DETR, Two-Stage Detectors
- **Language Models** -- GPT-1/2/3, BERT, Transformer
- **Reasoning Agents** -- ReAct + Reflexion + ReasoningBank (HotpotQA)
- **Segmentation** -- Segment Anything v2 (SAM2)
- **Face Detection** -- MTCNN + FaceNet pipeline
- **Vision Preprocessing** -- CV Preprocessing, Power Retention

Datasets used: COCO128, CIFAR-10, UTKFace, HotpotQA (auto-downloaded on first run).

---

## Module Overview

| Module                                      | Type       | Description                                          | Paper Review PDF                                                                          | Implementation PDF                                               |
| ------------------------------------------- | ---------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [Diffusion](#diffusion-1)                   | Code + PDF | GAN, VAE, DDPM, DDIM -- generative model progression | [Paper Review](./Diffusion/논문_삼석사와아이들_김재현_DiffusionModelPaperReview_1014.pdf) | [Implementation](./Diffusion/논문구현_Diffusion.pdf)             |
| [YOLO](#yolo-1)                             | Code + PDF | YOLOv1/v3/v5/v11 + MTCNN face detection              | [Paper Review](./YOLO/논문_삼석사와아이들_김재현_YoloPaperReview_1001.pdf)                | [Implementation](./YOLO/논문구현_Yolo.pdf)                       |
| [GPT](#gpt-1)                               | Code + PDF | GPT-1/2/3 + BERT -- Korean-English translation       | [Paper Review](./GPT/GPT_BERT_Paper_Review.pdf)                                           | [Implementation](./GPT/논문구현_GPTBERT.pdf)                     |
| [Transformer](#transformer-1)               | Code + PDF | Full "Attention is All You Need" implementation      | [Paper Review](./Transformer/Transformer_Paper_review.pdf)                                | [Implementation](./Transformer/논문구현_Transformer.pdf)         |
| [ReasoningBank](#reasoningbank-1)           | Code + PDF | ReAct + Reflexion agents with HotpotQA               | --                                                                                        | [Implementation](./ReasoningBank/논문구현_ReasoningBank.pdf)     |
| [SAM2](#sam2-1)                             | Code + PDF | Segment Anything v2 -- grid + interactive            | --                                                                                        | [Implementation](./SAM2/논문구현_SAM2.pdf)                       |
| [CV](#pdf-only-review-modules)              | PDF only   | Vision Preprocessing review                          | --                                                                                        | [Implementation](./CV/논문구현_vision_preprocessing.pdf)         |
| [Power_Retention](#pdf-only-review-modules) | PDF only   | Power Retention review                               | --                                                                                        | [Implementation](./Power_Retention/논문구현_Power_Retention.pdf) |
| [RT-DETR](#pdf-only-review-modules)         | PDF only   | Real-Time Detection Transformer review               | --                                                                                        | [Implementation](./RT-DETR/논문구현_RT-DETR.pdf)                 |
| [TwoStage](#pdf-only-review-modules)        | PDF only   | Two-Stage Detector review                            | --                                                                                        | [Implementation](./TwoStage/논문구현_Two-stage.pdf)              |

---

## Code Implementation Modules

### Diffusion

> Tracing the generative model evolution: GAN -> VAE -> DDPM -> DDIM

**PDF Documents:**

- [Paper Review -- Diffusion Model Paper Review](./Diffusion/논문_삼석사와아이들_김재현_DiffusionModelPaperReview_1014.pdf)
- [Implementation Document](./Diffusion/논문구현_Diffusion.pdf)

**Source Files:**

| File                                 | Description                                               |
| ------------------------------------ | --------------------------------------------------------- |
| [`GAN.py`](./Diffusion/GAN.py)       | Generative Adversarial Networks (Goodfellow et al., 2014) |
| [`VAE.py`](./Diffusion/VAE.py)       | Auto-Encoding Variational Bayes (Kingma & Welling, 2013)  |
| [`DDPM.py`](./Diffusion/DDPM.py)     | Denoising Diffusion Probabilistic Models                  |
| [`DDIM.py`](./Diffusion/DDIM.py)     | Denoising Diffusion Implicit Models (InstructPix2Pix)     |
| [`fixing.py`](./Diffusion/fixing.py) | Helper utilities                                          |

---

### YOLO

> Object detection from YOLOv1 to v11, plus MTCNN-based face detection pipeline

**PDF Documents:**

- [Paper Review -- YOLO Paper Review](./YOLO/논문_삼석사와아이들_김재현_YoloPaperReview_1001.pdf)
- [Implementation Document](./YOLO/논문구현_Yolo.pdf)

**Source Files:**

| File                                                | Description                     |
| --------------------------------------------------- | ------------------------------- |
| [`YOLOv1.py`](./YOLO/YOLOv1.py)                     | You Only Look Once v1 (COCO128) |
| [`YOLOv3.py`](./YOLO/YOLOv3.py)                     | YOLOv3 implementation           |
| [`YOLOv5.py`](./YOLO/YOLOv5.py)                     | YOLOv5 (with mAP evaluation)    |
| [`YOLOv11.py`](./YOLO/YOLOv11.py)                   | Ultralytics YOLOv11             |
| [`face_detecting.py`](./YOLO/face_detecting.py)     | MTCNN-based face detection      |
| [`face_crops.py`](./YOLO/face_crops.py)             | Face cropping pipeline          |
| [`face_detecting_2.py`](./YOLO/face_detecting_2.py) | Face detection v2               |
| [`face_detecting_3.py`](./YOLO/face_detecting_3.py) | Face detection v3               |

---

### GPT

> PyTorch implementations of GPT-1/2/3 and BERT, applied to Korean-English translation

**PDF Documents:**

- [Paper Review -- GPT & BERT Paper Review](./GPT/GPT_BERT_Paper_Review.pdf)
- [Implementation Document](./GPT/논문구현_GPTBERT.pdf)

**Source Files:**

| File                       | Description                                                        |
| -------------------------- | ------------------------------------------------------------------ |
| [`gpt.py`](./GPT/gpt.py)   | GPT-1: Improving Language Understanding by Generative Pre-Training |
| [`gpt2.py`](./GPT/gpt2.py) | GPT-2: Language Models are Unsupervised Multitask Learners         |
| [`gpt3.py`](./GPT/gpt3.py) | GPT-3: Language Models are Few-Shot Learners                       |
| [`bert.py`](./GPT/bert.py) | BERT: Pre-training of Deep Bidirectional Transformers              |

---

### Transformer

> Complete PyTorch implementation of "Attention Is All You Need" (Vaswani et al., 2017)

**PDF Documents:**

- [Paper Review -- Transformer Paper Review](./Transformer/Transformer_Paper_review.pdf)
- [Implementation Document](./Transformer/논문구현_Transformer.pdf)

**Source Files:**

| File                                 | Description                                                     |
| ------------------------------------ | --------------------------------------------------------------- |
| [`trans.py`](./Transformer/trans.py) | Multi-Head Attention, Positional Encoding, full Encoder-Decoder |

---

### ReasoningBank

> ReAct + Reflexion + ReasoningBank reasoning agents evaluated on HotpotQA with combination analysis

**PDF Documents:**

- [Implementation Document](./ReasoningBank/논문구현_ReasoningBank.pdf)

**Source Files:**

| File                                                                                       | Description                                     |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| [`react_reflexion_reasoningbank.py`](./ReasoningBank/react_reflexion_reasoningbank.py)     | Unified ReAct + Reflexion + ReasoningBank agent |
| [`hotpot_qa_with_llm.py`](./ReasoningBank/hotpot_qa_with_llm.py)                           | HotpotQA with OpenAI API                        |
| [`hotpot_qa_integration.py`](./ReasoningBank/hotpot_qa_integration.py)                     | HotpotQA integration pipeline                   |
| [`compare_combinations.py`](./ReasoningBank/compare_combinations.py)                       | Agent combination comparison                    |
| [`compare_all_combinations_hotpot.py`](./ReasoningBank/compare_all_combinations_hotpot.py) | All combinations comparison (HotpotQA)          |
| [`example_usage.py`](./ReasoningBank/example_usage.py)                                     | Usage examples                                  |
| [`hotpot.py`](./ReasoningBank/hotpot.py)                                                   | HotpotQA data loader                            |

**Dependency Files:**

| File                                                                 | Description             |
| -------------------------------------------------------------------- | ----------------------- |
| [`hotpot_requirements.txt`](./ReasoningBank/hotpot_requirements.txt) | HotpotQA experiments    |
| [`openai_requirements.txt`](./ReasoningBank/openai_requirements.txt) | OpenAI API integration  |
| [`llama_requirements.txt`](./ReasoningBank/llama_requirements.txt)   | LLaMA model integration |

**Analysis Results:**

- [comparison_table.md](./ReasoningBank/comparison_table.md) -- Agent combination performance comparison
- [hotpot_all_combinations_table.md](./ReasoningBank/hotpot_all_combinations_table.md) -- Full HotpotQA combination comparison

---

### SAM2

> Segment Anything Model v2 (Meta AI, 2024) -- grid-based auto and interactive segmentation

**PDF Documents:**

- [Implementation Document](./SAM2/논문구현_SAM2.pdf)

**Source Files:**

| File                                  | Description                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| [`sam2_test.py`](./SAM2/sam2_test.py) | Grid-based auto segmentation + point/box interactive segmentation |

---

## PDF-Only Review Modules

Modules containing only paper review documents without code implementations.

| Module              | Topic                                    | PDF Link                                                                        |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| **CV**              | Vision Preprocessing techniques          | [논문구현\_vision_preprocessing.pdf](./CV/논문구현_vision_preprocessing.pdf)    |
| **Power_Retention** | Power Retention mechanism                | [논문구현\_Power_Retention.pdf](./Power_Retention/논문구현_Power_Retention.pdf) |
| **RT-DETR**         | Real-Time Detection Transformer          | [논문구현\_RT-DETR.pdf](./RT-DETR/논문구현_RT-DETR.pdf)                         |
| **TwoStage**        | Two-Stage Detectors (Faster R-CNN, etc.) | [논문구현\_Two-stage.pdf](./TwoStage/논문구현_Two-stage.pdf)                    |

---

## Tech Stack

| Category             | Technologies                                   |
| -------------------- | ---------------------------------------------- |
| **Deep Learning**    | PyTorch, torchvision, torchaudio               |
| **Object Detection** | Ultralytics (YOLO), facenet-pytorch (MTCNN)    |
| **NLP / LLM**        | HuggingFace Transformers, accelerate, datasets |
| **API**              | OpenAI API                                     |
| **Computer Vision**  | OpenCV, PIL                                    |
| **Visualization**    | matplotlib                                     |
| **Utilities**        | numpy, tqdm, pandas, PyYAML, albumentations    |

---

## Getting Started

Every script is standalone and requires no CLI arguments.

```bash
# Create environment
conda create -n paperreview python=3.10
conda activate paperreview

# Install core packages
uv pip install torch torchvision torchaudio
uv pip install opencv-python matplotlib numpy tqdm pyyaml Pillow
uv pip install ultralytics facenet-pytorch albumentations

# Install NLP / LLM packages (GPT, Transformer, ReasoningBank)
uv pip install transformers accelerate datasets huggingface-hub openai pandas

# Run any script
python Diffusion/DDPM.py
python YOLO/YOLOv1.py
python GPT/gpt.py
python Transformer/trans.py
```

- Automatic CUDA/CPU detection
- Automatic dataset download (COCO128, CIFAR-10, UTKFace, HotpotQA)

---

## Project Structure

```
Paper-Review/
├── Diffusion/              # GAN, VAE, DDPM, DDIM implementations
│   ├── GAN.py
│   ├── VAE.py
│   ├── DDPM.py
│   ├── DDIM.py
│   ├── fixing.py
│   └── *.pdf
│
├── YOLO/                   # YOLOv1/v3/v5/v11 + face detection
│   ├── YOLOv1.py
│   ├── YOLOv3.py
│   ├── YOLOv5.py
│   ├── YOLOv11.py
│   ├── face_detecting.py
│   ├── face_crops.py
│   ├── face_detecting_2.py
│   ├── face_detecting_3.py
│   └── *.pdf
│
├── GPT/                    # GPT-1/2/3 + BERT
│   ├── gpt.py
│   ├── gpt2.py
│   ├── gpt3.py
│   ├── bert.py
│   ├── requirements.txt
│   └── *.pdf
│
├── Transformer/            # Attention is All You Need
│   ├── trans.py
│   └── *.pdf
│
├── ReasoningBank/          # ReAct + Reflexion agents
│   ├── react_reflexion_reasoningbank.py
│   ├── hotpot_qa_with_llm.py
│   ├── hotpot_qa_integration.py
│   ├── compare_combinations.py
│   ├── compare_all_combinations_hotpot.py
│   ├── example_usage.py
│   ├── hotpot.py
│   ├── *_requirements.txt
│   └── *.pdf
│
├── SAM2/                   # Segment Anything v2
│   ├── sam2_test.py
│   └── *.pdf
│
├── CV/                     # Vision Preprocessing (PDF only)
├── Power_Retention/        # Power Retention (PDF only)
├── RT-DETR/                # RT-DETR (PDF only)
└── TwoStage/               # Two-Stage Detector (PDF only)
```

---

## Code Conventions

- Each directory is **independent** -- no cross-directory imports
- Script structure: Imports -> Config -> Dataset -> Model -> Training -> Inference -> Visualization -> `__main__`
- All documentation and comments are written in **Korean**
- Visualizations use Korean-friendly font settings (Matplotlib + PIL)
- ReasoningBank outputs **structured JSON logs** per episode

---

## License

This repository is for **academic and educational purposes** only.
All code and documents are (c) 2025 Jaehyeon Kim. All rights reserved.
