# auto_face_label_and_predict.py
# ---------------------------------------------------------------
# 1) 입력 폴더의 이미지에서 얼굴 자동 검출/임베딩
# 2) 임베딩을 군집화하여 자동 라벨(클러스터) 부여
# 3) pseudo-label로 KNN 분류기 학습
# 4) 파일명이 'test'를 포함한 이미지들에서 얼굴 예측 및 시각화 저장
# ---------------------------------------------------------------

import os, re, math, shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import numpy

# ================== 사용자 설정 ==================
INPUT_DIR  = "./refs"            # 원본 이미지 폴더(스크린샷처럼 a1.jpg, b2.jpg, test3.jpg ...)
OUT_DIR    = "./refs/face_crops"  # 결과 저장 폴더
FACE_SIZE  = 160                   # FaceNet 기본 입력 크기
CONF_THRESH = 0.90                 # MTCNN face prob threshold
DIST_THRESHOLD = 0.35              # 군집화 거리 임계값(코사인). 작게 => 더 많은 클러스터
K_FOR_KNN = 3                      # KNN k
UNKNOWN_MAX_DIST = 0.45            # 예측 시 최근접 중심까지의 최대 허용 코사인 거리(크면 unknown 처리 줄어듦)
USE_GPU = torch.cuda.is_available()
# =================================================

def is_image(p: Path) -> bool:
    return p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]

def is_test_name(name: str) -> bool:
    return "test" in name.lower()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    # Pillow ≥ 8: textbbox 사용
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except AttributeError:
        # Fallback: font.getbbox / getmetrics + textlength
        try:
            l, t, r, b = font.getbbox(text)
            return (r - l, b - t)
        except AttributeError:
            w = draw.textlength(text, font=font)
            try:
                ascent, descent = font.getmetrics()
                h = ascent + descent
            except Exception:
                h = 18
            return (int(w), int(h))

def draw_boxes_and_labels(pil_img, boxes, labels):
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
        tw, th = _text_wh(draw, lab, font)
        draw.rectangle([(x1, y1 - th - 6), (x1 + tw + 8, y1)], fill=(0, 255, 0))
        draw.text((x1 + 4, y1 - th - 4), lab, fill=(0, 0, 0), font=font)
    return img

def draw_boxes_and_labels(pil_img: Image.Image, boxes: np.ndarray, labels: List[str]) -> Image.Image:
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    for (x1,y1,x2,y2), lab in zip(boxes, labels):
        draw.rectangle([(x1,y1),(x2,y2)], outline=(0,255,0), width=3)
        tw, th = draw.textsize(lab, font=font)
        draw.rectangle([(x1, y1- th - 6), (x1 + tw + 8, y1)], fill=(0,255,0))
        draw.text((x1+4, y1 - th - 4), lab, fill=(0,0,0), font=font)
    return img

def main():
    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"[INFO] device: {device}")

    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUT_DIR)
    faces_dir = out_dir / "faces_crops"
    vis_dir   = out_dir / "test_vis"
    ensure_dir(out_dir); ensure_dir(faces_dir); ensure_dir(vis_dir)

    # 모델 준비
    mtcnn = MTCNN(image_size=FACE_SIZE, margin=10, thresholds=[0.6, 0.7, 0.7],
                  post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 파일 목록 스캔
    all_imgs = [p for p in in_dir.iterdir() if p.is_file() and is_image(p)]
    train_imgs = [p for p in all_imgs if not is_test_name(p.name)]
    test_imgs  = [p for p in all_imgs if is_test_name(p.name)]
    print(f"[INFO] found {len(train_imgs)} train-like images, {len(test_imgs)} test images")

    # -----------------------------
    # 1) 얼굴 검출 + 임베딩 (train용)
    #    - 다인 사진은 가장 큰 얼굴 1개만 사용(노이즈 감소 목적)
    # -----------------------------
    train_recs = []  # dicts: {file, box, prob, emb(512)}
    print("[STEP] detect & embed (train)")
    for img_path in tqdm(train_imgs):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes)==0:
            continue
        # 가장 큰 얼굴 선택
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        if probs[idx] < CONF_THRESH:
            continue
        box = boxes[idx]
        # 정렬된 얼굴 텐서 얻기
        box_arr = np.expand_dims(box.astype(np.float32), axis=0)  # (1, 4) np.ndarray
        face = mtcnn.extract(img, box_arr, save_path=None)[0]
        # >>> add:
        if face.ndim == 3 and face.shape[0] == 1:   # grayscale -> RGB
            face = face.repeat(3, 1, 1)
        elif face.ndim == 2:                        # (H,W)인 경우까지 방어
            face = face.unsqueeze(0).repeat(3, 1, 1)

        ft = face.unsqueeze(0).to(device).float()
        with torch.no_grad():
            emb = resnet(ft).cpu().numpy()[0]  # 512-d
        train_recs.append({
            "file": img_path.name,
            "box": box.astype(np.float32),
            "prob": float(probs[idx]),
            "emb": emb
        })

    if len(train_recs) < 2:
        print("[ERROR] train 후보 얼굴이 너무 적습니다. 더 많은 단일 인물 이미지를 넣어주세요.")
        return

    # -----------------------------
    # 2) 임베딩 군집화 -> 자동 라벨
    # -----------------------------
    X = np.stack([r["emb"] for r in train_recs], axis=0)
    X = normalize(X)  # 코사인 거리 안정화

    print("[STEP] clustering (Agglomerative, cosine, threshold=%.3f)" % DIST_THRESHOLD)
    # sklearn 1.4+: metric='cosine' + linkage='average' 사용 가능
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=DIST_THRESHOLD,
                                        linkage='average', metric='cosine')
    labels = clusterer.fit_predict(X)  # 0..K-1
    n_clusters = int(labels.max()+1)
    print(f"[INFO] clusters: {n_clusters}")

    # 노이즈 라벨(-1)은 나오지 않지만, 혹시를 대비하여 별도 처리
    label_names = [f"person_{l:02d}" if l>=0 else "unknown" for l in labels]
    for rec, ln in zip(train_recs, label_names):
        rec["auto_label"] = ln

    # 라벨별 폴더에 크롭 저장
    for rec in train_recs:
        lab = rec["auto_label"]
        lab_dir = faces_dir / lab
        ensure_dir(lab_dir)
        # 크롭 저장
        try:
            img = Image.open(in_dir / rec["file"]).convert("RGB")
            x1,y1,x2,y2 = rec["box"]
            crop = img.crop((max(0,int(x1)), max(0,int(y1)), int(x2), int(y2))).resize((FACE_SIZE, FACE_SIZE))
            crop.save(lab_dir / rec["file"])
        except:
            pass

    # 매핑 CSV 저장
    df_train = pd.DataFrame({
        "file":[r["file"] for r in train_recs],
        "auto_label":[r["auto_label"] for r in train_recs],
        "x1":[r["box"][0] for r in train_recs],
        "y1":[r["box"][1] for r in train_recs],
        "x2":[r["box"][2] for r in train_recs],
        "y2":[r["box"][3] for r in train_recs],
        "prob":[r["prob"] for r in train_recs],
    })
    df_train.to_csv(out_dir / "autolabel_train.csv", index=False)

    # -----------------------------
    # 3) pseudo-label로 분류기 학습 (KNN)
    # -----------------------------
    y = np.array(label_names)
    knn = KNeighborsClassifier(n_neighbors=K_FOR_KNN, metric='cosine', weights='distance')
    knn.fit(X, y)

    # 각 클래스의 중심(centroid)도 저장해서 unknown 판단에 활용
    centroids = {}
    for lab in sorted(set(y)):
        emb_lab = X[y==lab]
        centroids[lab] = emb_lab.mean(axis=0)
        centroids[lab] = centroids[lab] / (np.linalg.norm(centroids[lab])+1e-8)

    # -----------------------------
    # 4) test 이미지 예측(여러 얼굴 모두 예측) + 시각화
    # -----------------------------
    test_rows = []
    print("[STEP] predict on test images")
    for img_path in tqdm(test_imgs):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        boxes, probs = mtcnn.detect(img)
        if boxes is None or len(boxes)==0:
            continue

        keep_boxes, keep_labels = [], []
        for b, p in zip(boxes, probs):
            if p < CONF_THRESH:
                continue
            face = mtcnn.extract(img, b[None, :].astype(np.float32), save_path=None)[0]
            if face.ndim == 3 and face.shape[0] == 1:
                face = face.repeat(3, 1, 1)
            elif face.ndim == 2:
                face = face.unsqueeze(0).repeat(3, 1, 1)
            ft = face.unsqueeze(0).to(device).float()
            with torch.no_grad():
                emb = resnet(ft).cpu().numpy()
            emb = normalize(emb)[0:1]

            # KNN 예측 + 최근접 중심 거리로 unknown 처리
            pred = knn.predict(emb)[0]
            proba = None
            try:
                # 거리 가중치라 predict_proba가 확률 비슷하게 나옴(참고용)
                proba = float(knn.predict_proba(emb).max())
            except:
                proba = 1.0

            # 최근접 중심까지의 코사인 거리
            d = cosine_distances(emb, centroids[pred].reshape(1,-1))[0,0]
            if d > UNKNOWN_MAX_DIST:
                pred_label = "unknown"
            else:
                pred_label = f"{pred} (d={d:.2f})"

            keep_boxes.append(b.astype(float))          # or np.asarray(b, dtype=float)
            keep_labels.append(pred_label)

        if keep_labels:
            vis = draw_boxes_and_labels(img, keep_boxes, keep_labels)
            vis.save(vis_dir / img_path.name)

    if len(test_rows)==0:
        print("[WARN] test 얼굴이 검출되지 않았습니다.")
    else:
        pd.DataFrame(test_rows).to_csv(out_dir / "test_predictions.csv", index=False)

    # 요약 출력
    print("\n[SUMMARY]")
    print(f"- 자동 라벨 개수: {n_clusters}")
    print(f"- 라벨 매핑 CSV: {out_dir/'autolabel_train.csv'}")
    print(f"- test 시각화:    {vis_dir}/")
    print(f"- test 결과 CSV:  {out_dir/'test_predictions.csv'}")
    print("\n[NOTE]")
    print(f"* 군집화 임계값 DIST_THRESHOLD={DIST_THRESHOLD}, unknown 판정 UNKNOWN_MAX_DIST={UNKNOWN_MAX_DIST}")
    print("  값이 너무 작으면 동일 인물이 쪼개지고, 크면 다른 사람이 합쳐질 수 있습니다.")
    print("  결과 보고 살짝 조정해 보세요.")

if __name__ == "__main__":
    main()
