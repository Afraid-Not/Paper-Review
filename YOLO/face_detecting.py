# yolo11_face_cls_and_predict.py
# -------------------------------------------------------------
# (A) faces_crops/ (사람이름별 폴더) -> YOLOv11 분류 데이터셋 자동 생성(train/val split)
# (B) YOLOv11 분류 모델 파인튜닝
# (C) 원본 사진들에서 MTCNN으로 얼굴 검출 -> YOLOv11-CLS로 신원 분류 -> 시각화 저장
# -------------------------------------------------------------

import os, random, shutil, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN

# ================== 경로/하이퍼파라미터 ==================
# faces_crops 스샷 기준 경로(폴더명은 각 사람 이름)
CROPS_DIR  = r"D:\D_Study\refs\face_crops"
# 원본/테스트 이미지가 있는 폴더(여기서 'test' 포함 파일들을 주로 시각화)
TEST_DIR   = r"D:\D_Study\refs\test"
# 출력 폴더
OUT_DIR    = r"D:\D_Study\refs\_out\yolo11_cls_out"

VAL_RATIO  = 0.2
EPOCHS     = 100
BATCH      = 16
IMG_SIZE   = 224
CONF_FACE  = 0.90  # MTCNN 얼굴 확신도 임계값
KEEP_TOPK  = None  # None이면 모든 얼굴 사용, 정수면 상위 K개만

# 가벼운 사전학습 가중치 우선
CLS_WEIGHTS_CANDIDATES = ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolov8n-cls.pt"]

# =========================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_split_from_crops(crops_dir: Path, dst_root: Path, val_ratio=0.2, seed=42):
    """faces_crops/이름별 폴더 -> dst_root/{train,val}/이름/ 이미지 복사"""
    random.seed(seed)
    train_root = dst_root / "train"
    val_root   = dst_root / "val"
    for d in [train_root, val_root]:
        ensure_dir(d)

    classes = []
    for person_dir in sorted([p for p in crops_dir.iterdir() if p.is_dir()]):
        cls_name = person_dir.name
        classes.append(cls_name)
        imgs = [p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
        if not imgs:
            continue
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs)*val_ratio)) if len(imgs) >= 5 else 1 if len(imgs)>=2 else 0
        val_imgs = imgs[:n_val]
        trn_imgs = imgs[n_val:]

        for split, items in [("train", trn_imgs), ("val", val_imgs)]:
            dst = (dst_root / split / cls_name)
            ensure_dir(dst)
            for im in items:
                shutil.copy2(im, dst / im.name)
    return classes

def pick_weights():
    for w in CLS_WEIGHTS_CANDIDATES:
        try:
            _ = YOLO(w)
            return w
        except Exception:
            continue
    # 랜덤 초기화
    return "yolo11n-cls.pt"

def latest_run_dir(root: Path) -> Path:
    runs = list((root/"runs"/"classify").glob("*"))
    if not runs: 
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)

def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except:
        return ImageFont.load_default()

def text_wh(draw, text, font):
    try:
        l,t,r,b = draw.textbbox((0,0), text, font=font)
        return r-l, b-t
    except:
        w = draw.textlength(text, font=font)
        try:
            asc, desc = font.getmetrics()
            h = asc + desc
        except:
            h = 18
        return int(w), int(h)

def draw_boxes_and_labels(pil_img: Image.Image, boxes: np.ndarray, labels: List[str]) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    drw = ImageDraw.Draw(img)
    fnt = load_font()
    for (x1,y1,x2,y2), lab in zip(boxes, labels):
        drw.rectangle([(x1,y1),(x2,y2)], outline=(0,255,0), width=3)
        tw, th = text_wh(drw, lab, fnt)
        drw.rectangle([(x1, y1-th-6), (x1+tw+8, y1)], fill=(0,255,0))
        drw.text((x1+4, y1-th-4), lab, fill=(0,0,0), font=fnt)
    return img

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    crops_dir = Path(CROPS_DIR)
    test_dir  = Path(TEST_DIR)
    out_dir   = Path(OUT_DIR)
    data_dir  = out_dir / "cls_data"
    vis_dir   = out_dir / "test_vis"
    for d in [out_dir, data_dir, vis_dir]:
        ensure_dir(d)

    # A) 데이터셋 생성
    print("[STEP] Build classification dataset from faces_crops")
    classes = make_split_from_crops(crops_dir, data_dir, VAL_RATIO)
    print(f"[INFO] classes ({len(classes)}): {classes}")

    # B) 학습
    print("[STEP] Train YOLOv11-CLS")
    weights = pick_weights()
    model = YOLO(weights)
    model.train(
        data=str(data_dir), 
        epochs=EPOCHS, imgsz=IMG_SIZE, 
        batch=BATCH, device=device,
        workers=0,  # Windows 안정
        lr0=0.001, patience=20, verbose=True
    )

    # best 가중치 로드
    run_dir = latest_run_dir(Path.cwd())
    if run_dir is None:
        raise RuntimeError("학습 결과(run dir)를 찾을 수 없습니다.")
    best = run_dir / "weights" / "best.pt"
    print(f"[INFO] best weights: {best}")
    cls_model = YOLO(str(best))

    # C) 테스트 이미지에서 MTCNN 검출 -> 분류 -> 시각화
    print("[STEP] Detect & Classify faces on test images")
    mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, device=device if device!="cpu" else None)
    test_imgs = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
    # 'test' 포함을 우선
    test_imgs = sorted(test_imgs, key=lambda p: ("test" not in p.name.lower(), p.name.lower()))

    total_faces = 0
    for img_path in tqdm(test_imgs):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes)==0:
            continue

        # 확신도 필터 + 상위 K 선택(옵션)
        keep = [(b, p) for b,p in zip(boxes, probs) if p is None or p >= CONF_FACE]
        if not keep:
            continue
        keep = sorted(keep, key=lambda x: ( (x[0][2]-x[0][0])*(x[0][3]-x[0][1]) ), reverse=True)
        if KEEP_TOPK is not None:
            keep = keep[:KEEP_TOPK]

        keep_boxes, keep_labels = [], []
        for b, p in keep:
            x1,y1,x2,y2 = [int(v) for v in b]
            crop = img.crop((max(0,x1), max(0,y1), x2, y2)).resize((IMG_SIZE, IMG_SIZE))
            # 분류 예측
            res = cls_model.predict(source=crop, imgsz=IMG_SIZE, verbose=False)[0]
            top1 = int(res.probs.top1)
            names = res.names if hasattr(res, "names") else cls_model.names
            name  = names[top1] if names and top1 in range(len(names)) else str(top1)
            conf  = float(res.probs.top1conf)
            keep_boxes.append(np.array([x1,y1,x2,y2], dtype=float))
            keep_labels.append(f"{name} ({conf:.2f})")
            total_faces += 1

        if keep_labels:
            vis = draw_boxes_and_labels(img, np.stack(keep_boxes,0), keep_labels)
            vis.save(vis_dir / img_path.name)

    print("\n[SUMMARY]")
    print(f"- Classes: {classes}")
    print(f"- Runs dir: {run_dir}")
    print(f"- Best weights: {best}")
    print(f"- Test visualizations: {vis_dir}")
    print(f"- Total faces annotated: {total_faces}")

if __name__ == "__main__":
    main()
