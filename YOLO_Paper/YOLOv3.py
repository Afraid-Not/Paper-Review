# train_yolov3_coco128_team.py
# -*- coding: utf-8 -*-
"""
YOLOv3 (Ultralytics) 학습/검증/예측 저장 + GT/Pred 좌우 합성 저장
- 모든 입출력: ./team_project/
"""

import os, zipfile, urllib.request, shutil, random, time
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import torch

# ===================== 사용자 설정 =====================
EPOCHS = 5
BATCH_SIZE = 16
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.70
SEED = 42

# ===================== 경로(모두 ./team_project/) =====================
ROOT = Path(__file__).resolve().parent
TP = ROOT / "team_project"; TP.mkdir(parents=True, exist_ok=True)

DATA_ROOT = TP / "_data"
COCO128_ZIP = DATA_ROOT / "coco128.zip"
COCO128_DIR = DATA_ROOT / "coco128"
IMAGES_DIR = COCO128_DIR / "images" / "train2017"
LABELS_DIR = COCO128_DIR / "labels" / "train2017"
DATA_YAML = DATA_ROOT / "coco128.yaml"

RUNS_DIR = TP / "_runs_yolov3"
OUT_DIR = TP / "_out_yolov3"
PRED_IMG_DIR = OUT_DIR / "pred_images"        # Ultralytics가 그려준 예측 이미지
CSV_DIR = OUT_DIR / "pred_csv"
# ▶ 커스텀 저장(우리 손으로 그림): GT/Pred/좌우합성
GT_DIR   = OUT_DIR / "gt"
PREDC_DIR= OUT_DIR / "pred_custom"
PAIRED_DIR=OUT_DIR / "paired"

for d in [OUT_DIR, PRED_IMG_DIR, CSV_DIR, GT_DIR, PREDC_DIR, PAIRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===================== 클래스 이름 =====================
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# ===================== 유틸 =====================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def download_coco128():
    if IMAGES_DIR.exists() and LABELS_DIR.exists():
        print(f"[Skip] COCO128 already exists: {COCO128_DIR}"); return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    print(f"[Download] {url}")
    with urllib.request.urlopen(url) as r, open(COCO128_ZIP, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[Extract] {COCO128_ZIP} -> {DATA_ROOT}")
    with zipfile.ZipFile(COCO128_ZIP, "r") as z:
        z.extractall(DATA_ROOT)

def write_coco128_yaml():
    data = {
        "path": str(COCO128_DIR.resolve()),
        "train": "images/train2017",
        "val":   "images/train2017",
        "names": {i:n for i,n in enumerate(COCO80)}
    }
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[DATA] yaml written: {DATA_YAML}")

def read_gt_yolo(txt_path: Path, w: int, h: int):
    boxes, labels = [], []
    if not txt_path.exists(): return boxes, labels
    for line in open(txt_path, "r").read().splitlines():
        if not line.strip(): continue
        cid, cx, cy, bw, bh = map(float, line.split())
        x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
        boxes.append((x1,y1,x2,y2))
        labels.append(COCO80[int(cid)])
    return boxes, labels

def draw_boxes(img: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str], scores=None, is_gt=False):
    out = img.copy()
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        color = (0,255,0) if is_gt else (255,0,0)   # GT=green, Pred=blue
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        txt = labels[i]
        if scores is not None: txt += f" {scores[i]:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yy = max(0, y1 - th - 4)
        cv2.rectangle(out, (x1, yy), (x1+tw+4, yy+th+4), color, -1)
        cv2.putText(out, txt, (x1+2, yy+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:left.shape[1]+right.shape[1]] = right
    return canvas

# ===================== 메인 =====================
def main():
    set_seed(SEED)
    download_coco128()
    write_coco128_yaml()

    from ultralytics import YOLO
    model = YOLO("yolov3u.pt")

    ultra_device = 0 if torch.cuda.is_available() else "cpu"

    # ---- 학습 ----
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        device=ultra_device,
        project=str(RUNS_DIR),
        name="train_coco128",
        exist_ok=True,
        verbose=True
    )

    # ---- 검증(mAP) ----
    val_res = model.val(
        data=str(DATA_YAML), imgsz=IMG_SIZE, conf=0.001, iou=IOU_THRES,
        split="val", device=ultra_device, project=str(RUNS_DIR),
        name="val_coco128", exist_ok=True, verbose=False
    )
    try:
        mAP_50_95 = float(val_res.box.map); mAP_50 = float(val_res.box.map50)
    except Exception:
        P,R,mAP_50,mAP_50_95 = val_res.mean_results()
    print(f"[YOLOv3] mAP@50-95: {mAP_50_95:.4f}")
    print(f"[YOLOv3] mAP@50    : {mAP_50:.4f}")

    # ---- 예측 저장(ultralytics 기본 저장) ----
    pred_results = model.predict(
        source=str(IMAGES_DIR), imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
        device=ultra_device, save=True, save_txt=True, save_conf=True,
        project=str(OUT_DIR), name="pred_images", exist_ok=True, verbose=False
    )
    pred_dir = Path(pred_results[0].save_dir) if isinstance(pred_results, list) and len(pred_results) else PRED_IMG_DIR
    # 라벨(txt) 폴더 추정
    pred_txt_dir = pred_dir / "labels"
    if not pred_txt_dir.exists():
        alt = pred_dir.parent / (pred_dir.name + "labels")
        pred_txt_dir = alt if alt.exists() else pred_txt_dir

    # ---- GT/Pred 좌우 합성 저장 (우리 손으로 그리기) ----
    for r in tqdm(pred_results, desc="Save GT/Pred paired"):
        img_path = Path(r.path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT
        gt_boxes, gt_labels = read_gt_yolo(LABELS_DIR / f"{img_path.stem}.txt", w, h)
        img_gt = draw_boxes(img, gt_boxes, gt_labels, is_gt=True)

        # Pred
        pred_boxes, pred_labels, pred_scores = [], [], []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
                pred_boxes.append((int(x1),int(y1),int(x2),int(y2)))
                pred_labels.append(COCO80[k] if 0 <= k < len(COCO80) else str(k))
                pred_scores.append(float(c))
        img_pred = draw_boxes(img, pred_boxes, pred_labels, pred_scores, is_gt=False)

        # 저장 (개별 및 좌우 합성)
        stem = img_path.stem
        cv2.imwrite(str(GT_DIR / f"{stem}_gt.jpg"), img_gt)
        cv2.imwrite(str(PREDC_DIR / f"{stem}_pred.jpg"), img_pred)
        pair = side_by_side(img_gt, img_pred)
        cv2.imwrite(str(PAIRED_DIR / f"{stem}_paired.jpg"), pair)

    # ---- CSV 내보내기 ----
    rows = []
    for r in pred_results:
        p = Path(r.path)
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        conf = r.boxes.conf.cpu().numpy().tolist()
        cls  = r.boxes.cls.cpu().numpy().astype(int).tolist()
        for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
            rows.append({"image": p.name, "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "conf": float(c), "cls": int(k)})
    df = pd.DataFrame(rows, columns=["image","x1","y1","x2","y2","conf","cls"])
    csv_path = CSV_DIR / "predictions.csv"; df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\n✅ Done!")
    print(f" Data        : {COCO128_DIR}")
    print(f" Train runs  : {RUNS_DIR}")
    print(f" Pred imgs   : {pred_dir}")
    print(f" Pred txt    : {pred_txt_dir if pred_txt_dir.exists() else '(none)'}")
    print(f" CSV         : {csv_path}")
    print(f" GT imgs     : {GT_DIR}")
    print(f" Pred imgs*  : {PREDC_DIR}")
    print(f" Paired imgs : {PAIRED_DIR}")

if __name__ == "__main__":
    main()
