# yolo11_face_cls_aug_and_predict.py
# -------------------------------------------------------------
# faces_crops/<이름>/*  -> 증강 생성 + train/val split
# YOLOv11 분류 학습 -> 원본 사진에서 MTCNN 검출 후 분류/시각화
# CUDA 사용 강화 + 한글 폰트 문제 해결 버전
# -------------------------------------------------------------
import os, random, shutil, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy

import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import albumentations as A

import matplotlib
from matplotlib import font_manager, rcParams

# ==================== CUDA 체크 ====================
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("⚠️ CUDA NOT AVAILABLE - Will use CPU")
print("=" * 60)
# ==================================================

# 한글 폰트 경로 리스트
KOREAN_FONTS = [
    r"C:/Windows/Fonts/malgun.ttf",                          # Windows - 맑은고딕
    r"C:/Windows/Fonts/malgunbd.ttf",                        # Windows - 맑은고딕 볼드
    r"C:/Windows/Fonts/gulim.ttc",                           # Windows - 굴림
    r"C:/Windows/Fonts/batang.ttc",                          # Windows - 바탕
    r"C:/Windows/Fonts/NanumGothic.ttf",                     # 나눔고딕
    r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",      # Ubuntu
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    r"/System/Library/Fonts/AppleSDGothicNeo.ttc",           # macOS
    "./NanumGothic.ttf",                                     # 프로젝트 폴더
]

# 전역 변수로 폰트 경로 저장
FOUND_FONT_PATH = None

def _set_korean_font():
    """Matplotlib 한글 폰트 설정"""
    global FOUND_FONT_PATH
    font_path = next((p for p in KOREAN_FONTS if os.path.exists(p)), None)
    if font_path:
        FOUND_FONT_PATH = font_path
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rcParams["font.family"] = font_name
        rcParams["axes.unicode_minus"] = False
        try:
            cachedir = matplotlib.get_cachedir()
            for f in os.listdir(cachedir):
                if f.startswith("fontlist") and f.endswith(".json"):
                    os.remove(os.path.join(cachedir, f))
        except Exception:
            pass
        print(f"[FONT] Matplotlib & PIL font set to: {font_name} ({font_path})")
    else:
        print("=" * 60)
        print("⚠️ 한글 폰트를 찾지 못했습니다!")
        print("다음 중 하나를 설치하세요:")
        print("1. 나눔고딕: https://hangeul.naver.com/font")
        print("2. 또는 'NanumGothic.ttf' 파일을 프로젝트 폴더에 복사")
        print("=" * 60)

_set_korean_font()

# ================== 경로/하이퍼파라미터 ==================
CROPS_DIR  = r"D:/D_Study/refs/faces_crops"
TEST_DIR   = r"D:/D_Study/refs/test"
OUT_DIR    = r"D:/D_Study/refs/_out/y11_cls_aug_out"

VAL_RATIO  = 0.2
EPOCHS     = 20
BATCH      = 16
IMG_SIZE   = 224
CONF_FACE  = 0.90
KEEP_TOPK  = None
SEED       = 42

BALANCE_MODE = "max"
TARGET_PER_CLASS = 300
MAX_AUG_PER_SRC = 20
JPEG_QUALITY_RANGE = (60, 95)

CLS_WEIGHTS = ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolov8n-cls.pt"]
# =========================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_font(size=20):
    """한글 지원 폰트 로드 (PIL용)"""
    global FOUND_FONT_PATH
    if FOUND_FONT_PATH and os.path.exists(FOUND_FONT_PATH):
        try:
            return ImageFont.truetype(FOUND_FONT_PATH, size)
        except Exception as e:
            print(f"[WARN] 한글 폰트 로드 실패: {e}")
    
    # 폴백: 시스템 폰트 재탐색
    for font_path in KOREAN_FONTS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # 최종 폴백
    print("[WARN] 한글 폰트를 찾지 못해 기본 폰트 사용 (한글 깨짐)")
    return ImageFont.load_default()

def text_wh(draw, text, font):
    try:
        l,t,r,b = draw.textbbox((0,0), text, font=font)
        return r-l, b-t
    except:
        w = draw.textlength(text, font=font)
        try:
            asc, desc = font.getmetrics()
            h = asc+desc
        except:
            h = 18
        return int(w), int(h)

def draw_boxes_and_labels(pil_img: Image.Image, boxes: np.ndarray, labels: List[str]) -> Image.Image:
    """한글 이름이 포함된 바운딩 박스 그리기"""
    img = pil_img.convert("RGB").copy()
    drw = ImageDraw.Draw(img)
    fnt = load_font(size=24)  # 크기 증가
    
    for (x1,y1,x2,y2), lab in zip(boxes, labels):
        # 박스 그리기
        drw.rectangle([(x1,y1),(x2,y2)], outline=(0,255,0), width=3)
        
        # 텍스트 배경 박스
        tw, th = text_wh(drw, lab, fnt)
        drw.rectangle([(x1, y1-th-8), (x1+tw+10, y1)], fill=(0,255,0))
        
        # 텍스트 그리기
        drw.text((x1+5, y1-th-6), lab, fill=(0,0,0), font=fnt)
    
    return img

def build_aug_pipeline(img_size=224):
    """얼굴 분류에 유효한 강·약 혼합 증강 (Albumentations 최신 API 호환)"""
    return A.Compose([
        A.OneOf([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC, p=1.0),
        ], p=1.0),

        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.8),

        A.OneOf([
            A.ColorJitter(0.25, 0.25, 0.2, 0.1, p=1.0),
            A.RandomBrightnessContrast(0.25, 0.2, p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
            A.RGBShift(10,10,10, p=1.0),
        ], p=0.8),

        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
        ], p=0.5),

        A.CoarseDropout(max_holes=2,
                        max_height=int(img_size*0.12), max_width=int(img_size*0.12),
                        min_holes=1, fill_value=0, p=0.35),

        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.6),

        A.Downscale(scale_min=0.85, scale_max=0.95, p=0.3),
    ])

def make_split_from_crops(crops_dir: Path, dst_root: Path, val_ratio=0.2, seed=42):
    """faces_crops/이름별 폴더 -> dst_root/{train,val}/이름/ 이미지 복사"""
    random.seed(seed)
    train_root = dst_root / "train"; val_root = dst_root / "val"
    ensure_dir(train_root); ensure_dir(val_root)
    classes = []
    for person_dir in sorted([p for p in crops_dir.iterdir() if p.is_dir()]):
        cls_name = person_dir.name; classes.append(cls_name)
        imgs = [p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
        if not imgs: continue
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs)*val_ratio)) if len(imgs)>=5 else (1 if len(imgs)>=2 else 0)
        val_imgs = imgs[:n_val]; trn_imgs = imgs[n_val:]
        for split, items in [("train", trn_imgs), ("val", val_imgs)]:
            dst = (dst_root / split / cls_name); ensure_dir(dst)
            for im in items: shutil.copy2(im, dst / im.name)
    return classes

def count_per_class(root: Path) -> dict:
    counts = {}
    for cls_dir in (root).iterdir():
        if not cls_dir.is_dir(): continue
        n = len([p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])
        counts[cls_dir.name] = n
    return counts

def augment_and_balance(train_root: Path, img_size=224, mode="max", target_per_class=300,
                        max_aug_per_src=20, seed=42):
    """train/<cls>/* 에 증강을 생성하여 클래스별 갯수를 맞춤"""
    random.seed(seed); np.random.seed(seed)
    aug = build_aug_pipeline(img_size)
    cls_counts = count_per_class(train_root)
    if not cls_counts: return {}
    if mode == "max":
        target = max(cls_counts.values())
        target = max(target, min(target_per_class, 200))
    elif mode == "fixed":
        target = target_per_class
    else:
        target = max(cls_counts.values())

    print(f"[AUG] class counts before: {cls_counts}")
    print(f"[AUG] target per class: {target}")

    for cls_name, n_now in cls_counts.items():
        need = max(0, target - n_now)
        if need == 0: continue

        cls_dir = train_root / cls_name
        base_imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
        if not base_imgs: continue

        created = 0
        tries = 0
        while created < need and tries < need * 3:
            tries += 1
            src = random.choice(base_imgs)
            base_prefix = src.stem
            existing = len(list(cls_dir.glob(base_prefix + "_aug_*.*")))
            if existing >= max_aug_per_src: 
                continue

            img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            auged = aug(image=img)["image"]
            out_name = f"{src.stem}_aug_{int(time.time()*1000)}_{random.randint(0,9999)}.jpg"
            out_path = cls_dir / out_name
            cv2.imencode(".jpg", auged, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(*JPEG_QUALITY_RANGE)])[1].tofile(str(out_path))
            created += 1

    cls_counts_after = count_per_class(train_root)
    print(f"[AUG] class counts after:  {cls_counts_after}")
    return cls_counts_after

def pick_weights():
    for w in CLS_WEIGHTS:
        try:
            _ = YOLO(w)
            return w
        except Exception:
            continue
    return "yolo11n-cls.pt"

def latest_run_dir(root: Path) -> Path:
    runs = list((root/"runs"/"classify").glob("*"))
    if not runs: return None
    return max(runs, key=lambda p: p.stat().st_mtime)

def main():
    random.seed(SEED); np.random.seed(SEED)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(0)
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[DEVICE] Using CPU")

    crops_dir = Path(CROPS_DIR)
    test_dir  = Path(TEST_DIR)
    out_dir   = Path(OUT_DIR)
    data_dir  = out_dir / "cls_data"
    vis_dir   = out_dir / "test_vis"
    for d in [out_dir, data_dir, vis_dir]:
        ensure_dir(d)

    # 1) Split
    print("[STEP] Build classification dataset from faces_crops")
    classes = make_split_from_crops(crops_dir, data_dir, VAL_RATIO, seed=SEED)
    print(f"[INFO] classes ({len(classes)}): {classes}")

    # 2) Augment (train만)
    print("[STEP] Augment & Balance train set")
    train_root = data_dir / "train"
    augment_and_balance(train_root, img_size=IMG_SIZE, mode=BALANCE_MODE,
                        target_per_class=TARGET_PER_CLASS, max_aug_per_src=MAX_AUG_PER_SRC, seed=SEED)

    # 3) Train YOLOv11-CLS
    print("[STEP] Train YOLOv11-CLS")
    weights = pick_weights()
    model = YOLO(weights)
    
    model.train(
        data=str(data_dir),
        epochs=EPOCHS, imgsz=IMG_SIZE,
        batch=BATCH, 
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,
        auto_augment="randaugment",
        erasing=0.25,
        lr0=0.001, patience=20, verbose=True,
        label_smoothing=0.1
    )

    run_dir = latest_run_dir(Path.cwd())
    if run_dir is None:
        raise RuntimeError("학습 결과(run dir)를 찾지 못했습니다.")
    best = run_dir / "weights" / "best.pt"
    print(f"[INFO] best weights: {best}")
    cls_model = YOLO(str(best))

    # 4) Test: MTCNN 검출 -> 분류 -> 시각화
    print("[STEP] Detect & Classify faces on test images")
    
    if torch.cuda.is_available():
        mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, device='cuda:0')
        print("[MTCNN] Using CUDA")
    else:
        mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, post_process=True, device='cpu')
        print("[MTCNN] Using CPU")
    
    test_imgs = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]]
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
        keep = [(b,p) for b,p in zip(boxes, probs) if p is None or p >= CONF_FACE]
        if not keep: 
            continue
        keep = sorted(keep, key=lambda x: ((x[0][2]-x[0][0])*(x[0][3]-x[0][1])), reverse=True)
        if KEEP_TOPK is not None:
            keep = keep[:KEEP_TOPK]

        keep_boxes, keep_labels = [], []
        for b, p in keep:
            x1,y1,x2,y2 = [int(v) for v in b]
            crop = img.crop((max(0,x1), max(0,y1), x2, y2)).resize((IMG_SIZE, IMG_SIZE))
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
    print(f"- Best weights: {best}")
    print(f"- Test visualizations: {vis_dir}")
    print(f"- Total faces annotated: {total_faces}")

if __name__ == "__main__":
    main()