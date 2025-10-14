# 🧠 Paper Review Repository
> Deep Learning 연구 논문 구현 및 재현 아카이브  
> 김재현 (Jaehyeon Kim) | AI Research & Implementation Study

---

## 📂 Repository Overview

이 저장소는 **생성 모델(GAN → VAE → Diffusion)** 과 **객체 탐지 모델(YOLOv1 → YOLOv11)** 의 논문을 직접 읽고  
PyTorch 기반으로 **구현·재현·시각화**한 결과를 정리한 개인 스터디 리포지토리입니다.  

각 모델별로 논문을 분석하며 원리를 코드로 옮기고,  
필요한 경우 COCO128 / CIFAR10 / UTKFace 등의 공개 데이터셋을 활용해 실험을 진행했습니다.

---

## 🧩 Directory Structure

```
paper_review/
├── Diffusion_Paper/
│   ├── GAN.py
│   ├── VAE.py
│   ├── DDPM.py
│   ├── DDIM.py
│   ├── face_aging_results.png
│   ├── 논문_삼성사학와아이트_김재현_DiffusionModelPaperReview_1014.pdf
│
├── YOLO_Paper/
│   ├── YOLOv1.py
│   ├── YOLOv3.py
│   ├── YOLOv5.py
│   ├── YOLOv11.py
│   ├── face_detecting*.py
│   ├── face_crops.py
│   ├── 논문_삼성사학와아이트_김재현_YoloPaperReview_1001.pdf
│
├── SAM2_Paper/
│   ├── sam2_test.py
│   ├── auto_result.png
│   ├── interactive_result.png
│
└── .gitignore
```

---

## 🚀 Implemented Papers

| 분야 | 모델 | 논문명 / 구현 내용 |
|------|------|--------------------|
| **Generative Models** | **GAN** | *Generative Adversarial Networks* (Goodfellow et al., 2014) |
| | **VAE** | *Auto-Encoding Variational Bayes* (Kingma & Welling, 2013) |
| | **DDPM / DDIM** | *Denoising Diffusion Probabilistic Models* / *Denoising Diffusion Implicit Models* |
| **Object Detection** | **YOLOv1–v11** | *You Only Look Once* (Redmon et al., 2016) → 최신 Ultralytics YOLOv11까지 |
| **Segmentation** | **SAM2** | *Segment Anything Model v2* (Meta AI, 2024) |
| **Face Recognition** | **MTCNN + FaceNet + YOLOv11-CLS** | 얼굴 검출 → 임베딩 → 자동 라벨링 → YOLO 분류 모델 파인튜닝 |

---

## 🧠 Key Features

- **End-to-End 구현**: 논문 구조를 PyTorch 코드로 완전 재현
- **모듈 단위 비교**: `GAN.py`, `VAE.py`, `DDPM.py` 등은 동일한 데이터로 결과 비교 가능
- **시각화 중심 학습**: 모든 실험 결과를 이미지(`*_results.png`)로 저장
- **자동화된 실험 파이프라인**: COCO128 자동 다운로드 + 결과 폴더 자동 정리
- **한글 친화적 시각화**: Matplotlib + PIL 폰트 설정으로 그래프/박스에 한글 지원

---

## 🧩 Example Results

| 모델 | 결과 예시 |
|------|------------|
| GAN | ![GAN CIFAR10](Diffusion_Paper/gan_cifar10_results.png) |
| VAE | ![VAE CIFAR10](Diffusion_Paper/vae_cifar10_results.png) |
| DDPM | ![Face Aging](Diffusion_Paper/face_aging_results.png) |
| YOLOv5 | ![COCO128 Detection](YOLO_Paper/sample_detection.png) |
| SAM2 | ![Segmentation](SAM2_Paper/auto_result.png) |

---

## ⚙️ Environment

```bash
# Base Environment
conda create -n paperreview python=3.10
conda activate paperreview

# Core Dependencies
pip install torch torchvision torchaudio
pip install opencv-python matplotlib numpy tqdm pyyaml
pip install ultralytics facenet-pytorch albumentations
```

---

## 🧭 Author

**김재현 (Jaehyeon Kim)**  
> AI Researcher / Developer  
> Focus: Deep Learning, Computer Vision, Generative Models, Representation Learning  
> 📍 Seoul, Republic of Korea  
> 📫 GitHub: [PlusMinusAnd](https://github.com/PlusMinusAnd)

---

## 🧾 License
This repository is for **academic and educational purposes** only.  
All code and assets are © 2025 Jaehyeon Kim unless otherwise stated.

---

> “읽고 → 이해하고 → 재현하고 → 설명한다.”  
> — 그게 나의 연구 루틴입니다.
