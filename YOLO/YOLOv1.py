# train_yolov1_coco128_team_project.py
# -*- coding: utf-8 -*-
"""
모든 입출력을 ./team_project/ 아래로 정리:
- ./team_project/_data/coco128 ... (자동 다운로드/압축해제)
- ./team_project/_out_yolov1/{gt,pred,paired} ... (시각화 결과)

YOLOv1 (간단 구현)로 COCO128(80클래스, YOLO txt 라벨) 학습/평가/시각화
필요 패키지(1회): pip install torch opencv-python numpy tqdm pyyaml
"""

import os, zipfile, urllib.request, shutil, random, math, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 설정 =====================
S = 7            # grid size
B = 2            # boxes per cell
C = 80           # COCO classes
IMG_SIZE = 448
EPOCHS = 30       # 데모용 (COCO128에선 2~5로 테스트, COCO2017은 늘리세요)
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 5e-4
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
CONF_THRES = 0.03
NMS_IOU = 0.5
MAP_IOU_THRESHOLDS = [x/100 for x in range(50, 100, 5)]  # 0.50:0.95
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 경로: 모두 ./team_project/ 아래 =====================
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "team_project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = PROJECT_DIR / "_data"
COCO128_ZIP = DATA_ROOT / "coco128.zip"
COCO128_DIR = DATA_ROOT / "coco128"
IMAGES_DIR = COCO128_DIR / "images" / "train2017"
LABELS_DIR = COCO128_DIR / "labels" / "train2017"

OUT_DIR = PROJECT_DIR / "_out_yolov1"
OUT_PRED = OUT_DIR / "pred"
OUT_GT   = OUT_DIR / "gt"
OUT_PAIR = OUT_DIR / "paired"

# ===================== 유틸 =====================
def set_seed(seed:int=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def download_coco128():
    if IMAGES_DIR.exists() and LABELS_DIR.exists():
        print(f"[Skip] COCO128 already exists at: {COCO128_DIR}")
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    print(f"[Download] {url}")
    with urllib.request.urlopen(url) as r, open(COCO128_ZIP, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[Extract] {COCO128_ZIP} -> {DATA_ROOT}")
    with zipfile.ZipFile(COCO128_ZIP, "r") as z:
        z.extractall(DATA_ROOT)

def list_images(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in exts])

def read_yolo_labels(txt_path: Path) -> List[Tuple[int,float,float,float,float]]:
    res = []
    if not txt_path.exists(): return res
    for line in open(txt_path, "r").read().splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) != 5: continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w  = min(max(w , 0.0), 1.0)
        h  = min(max(h , 0.0), 1.0)
        res.append((cid, cx, cy, w, h))
    return res

def resize_pad(img: np.ndarray, size: int = IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def draw_boxes(img_bgr: np.ndarray, boxes_xyxy: List[Tuple[int,int,int,int]],
               labels: List[str], scores: List[float] = None, gt: bool = False) -> np.ndarray:
    out = img_bgr.copy()
    for i,(x1,y1,x2,y2) in enumerate(boxes_xyxy):
        color = (0,255,0) if gt else (255,0,0)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        txt = labels[i]
        if scores is not None: txt += f" {scores[i]:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yy = max(0, y1 - th - 4)
        cv2.rectangle(out, (x1, yy), (x1+tw+4, yy+th+4), color, -1)
        cv2.putText(out, txt, (x1+2, yy+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def side_by_side(a: np.ndarray, b: np.ndarray):
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1]+b.shape[1], 3), dtype=np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1]] = b
    return canvas

def xywh_to_xyxy_norm(cx, cy, w, h):
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return x1, y1, x2, y2

def iou_xyxy(a: torch.Tensor, b: torch.Tensor):
    # a,b: (..., 4) in xyxy normalized [0,1]
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter = inter_w * inter_h
    area_a = (ax2-ax1).clamp(0) * (ay2-ay1).clamp(0)
    area_b = (bx2-bx1).clamp(0) * (by2-by1).clamp(0)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_j = (boxes[idxs[1:],2]-boxes[idxs[1:],0]) * (boxes[idxs[1:],3]-boxes[idxs[1:],1])  # ← 한 번만 계산
        iou = inter / (area_i + area_j - inter + 1e-9)
        idxs = idxs[1:][iou < iou_thres]
    return keep

# ===================== 데이터셋 =====================
class YoloTxtDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: Path, lab_dir: Path, img_size=IMG_SIZE, S=S, C=C, augment=True, limit=None):
        self.img_dir = img_dir; self.lab_dir = lab_dir
        self.img_paths = list_images(img_dir)
        if limit is not None: self.img_paths = self.img_paths[:limit]
        self.img_size = img_size; self.S = S; self.C = C; self.augment = augment

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lab_path = self.lab_dir / (img_path.stem + ".txt")
        img_bgr = cv2.imread(str(img_path))
        h0, w0 = img_bgr.shape[:2]

        # 간단 augmentation (hflip)
        flipped = False
        if self.augment and random.random() < 0.5:
            img_bgr = cv2.flip(img_bgr, 1)
            flipped = True

        img_bgr = resize_pad(img_bgr, self.img_size)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img_rgb.transpose(2,0,1))  # [3,H,W]

        # 타겟 생성
        obj = torch.zeros((self.S, self.S), dtype=torch.float32)
        box = torch.zeros((self.S, self.S, 4), dtype=torch.float32)  # (tx,ty,w,h) normalized
        cls = torch.full((self.S, self.S), -1, dtype=torch.int64)    # -1=noobj

        labels = read_yolo_labels(lab_path)
        for cid, cx, cy, w, h in labels:
            if flipped: cx = 1 - cx  # 좌우 반전
            gx = cx * self.S; gy = cy * self.S
            gi = min(self.S-1, int(gx)); gj = min(self.S-1, int(gy))
            if obj[gj, gi] == 1:  # 셀 충돌 시 더 큰 박스 우선
                prev_w, prev_h = box[gj, gi, 2].item(), box[gj, gi, 3].item()
                if w*h <= prev_w*prev_h: continue
            tx = gx - gi; ty = gy - gj
            obj[gj, gi] = 1.0
            box[gj, gi] = torch.tensor([tx, ty, w, h], dtype=torch.float32)
            cls[gj, gi] = cid

        meta = {"path": str(img_path), "orig_shape": (h0,w0)}
        return img, obj, box, cls, meta

# ===================== 모델(YOLOv1 간단형) =====================
class ConvBNL(nn.Module):
    def __init__(self, c_in, c_out, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, S=S, B=B, C=C):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.backbone = nn.Sequential(
            ConvBNL(3, 64, 7, 2, 3),   # 224
            nn.MaxPool2d(2,2),         # 112
            ConvBNL(64,128,3,1,1),
            nn.MaxPool2d(2,2),         # 56
            ConvBNL(128,256,3,1,1),
            nn.MaxPool2d(2,2),         # 28
            ConvBNL(256,512,3,1,1),
            nn.MaxPool2d(2,2),         # 14
            ConvBNL(512,1024,3,1,1),
        )
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),            # 7x7
            ConvBNL(1024,512,3,1,1),
        )
        self.head = nn.Conv2d(512, B*5 + C, 1, 1, 0) # 7x7x(10+C)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)                 # [N, 10+C, 7,7]
        x = x.permute(0,2,3,1).contiguous()  # [N,7,7,10+C]
        return x

# ===================== 손실함수 =====================
class YoloV1Loss(nn.Module):
    def __init__(self, S=S, B=B, C=C, l_coord=LAMBDA_COORD, l_noobj=LAMBDA_NOOBJ):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        self.register_buffer("gx", gx.float())
        self.register_buffer("gy", gy.float())

    def forward(self, pred, obj_mask, gt_box, gt_cls):
        N = pred.size(0)
        b0 = pred[...,:5]
        b1 = pred[...,5:10]
        cls_logits = pred[...,10:]

        def activate(b):
            tx = torch.sigmoid(b[...,0])
            ty = torch.sigmoid(b[...,1])
            tw = torch.sigmoid(b[...,2])
            th = torch.sigmoid(b[...,3])
            conf = torch.sigmoid(b[...,4])
            return tx,ty,tw,th,conf
        tx0,ty0,tw0,th0,conf0 = activate(b0)
        tx1,ty1,tw1,th1,conf1 = activate(b1)

        txg, tyg, twg, thg = gt_box.unbind(-1)  # (N,S,S)
        cxg = (txg + self.gx) / self.S
        cyg = (tyg + self.gy) / self.S
        x1g = cxg - twg/2; y1g = cyg - thg/2; x2g = cxg + twg/2; y2g = cyg + thg/2
        gt_xyxy = torch.stack([x1g,y1g,x2g,y2g], dim=-1).clamp(0,1)

        def pred_xyxy(tx,ty,tw,th):
            cx = (tx + self.gx) / self.S
            cy = (ty + self.gy) / self.S
            x1 = cx - tw/2; y1 = cy - th/2; x2 = cx + tw/2; y2 = cy + th/2
            return torch.stack([x1,y1,x2,y2], dim=-1).clamp(0,1)

        p0_xyxy = pred_xyxy(tx0,ty0,tw0,th0)
        p1_xyxy = pred_xyxy(tx1,ty1,tw1,th1)

        iou0 = iou_xyxy(p0_xyxy, gt_xyxy)
        iou1 = iou_xyxy(p1_xyxy, gt_xyxy)

        obj = obj_mask.bool()
        choose_b0 = (iou0 > iou1) & obj
        choose_b1 = (~choose_b0) & obj

        def coord_loss(tx,ty,tw,th, choose_mask):
            txg_s = txg[choose_mask]; tyg_s = tyg[choose_mask]
            twg_s = twg[choose_mask]; thg_s = thg[choose_mask]
            tx_s = tx[choose_mask]; ty_s = ty[choose_mask]
            tw_s = tw[choose_mask]; th_s = th[choose_mask]
            lxy = F.mse_loss(tx_s, txg_s, reduction="sum") + F.mse_loss(ty_s, tyg_s, reduction="sum")
            lwh = F.mse_loss(torch.sqrt(tw_s + 1e-9), torch.sqrt(twg_s + 1e-9), reduction="sum") + \
                  F.mse_loss(torch.sqrt(th_s + 1e-9), torch.sqrt(thg_s + 1e-9), reduction="sum")
            return lxy + lwh

        L_coord = coord_loss(tx0,ty0,tw0,th0, choose_b0) + coord_loss(tx1,ty1,tw1,th1, choose_b1)

        conf_t0 = torch.zeros_like(conf0); conf_t1 = torch.zeros_like(conf1)
        conf_t0[choose_b0] = iou0[choose_b0].detach()
        conf_t1[choose_b1] = iou1[choose_b1].detach()

        L_obj = F.mse_loss(conf0[choose_b0], conf_t0[choose_b0], reduction="sum") + \
                F.mse_loss(conf1[choose_b1], conf_t1[choose_b1], reduction="sum")

        noobj_mask0 = ~choose_b0
        noobj_mask1 = ~choose_b1
        L_noobj = F.mse_loss(conf0[noobj_mask0], torch.zeros_like(conf0[noobj_mask0]), reduction="sum") + \
                  F.mse_loss(conf1[noobj_mask1], torch.zeros_like(conf1[noobj_mask1]), reduction="sum")

        cls_prob = F.softmax(cls_logits, dim=-1)
        onehot = torch.zeros_like(cls_prob)
        valid = obj & (gt_cls>=0)
        if valid.any():
            onehot[valid] = F.one_hot(gt_cls[valid], num_classes=self.C).float()
        L_cls = F.mse_loss(cls_prob[valid], onehot[valid], reduction="sum") if valid.any() else torch.tensor(0.0, device=pred.device)

        loss = LAMBDA_COORD * L_coord + L_obj + LAMBDA_NOOBJ * L_noobj + L_cls
        return loss / max(1, int(obj.sum().item()))

# ===================== 학습/평가 =====================
def train_one_epoch(model, crit, loader, opt):
    model.train()
    total = 0.0
    for img, obj, box, cls, _ in tqdm(loader, desc="Train", leave=False):
        img = img.to(DEVICE); obj=obj.to(DEVICE); box=box.to(DEVICE); cls=cls.to(DEVICE)
        pred = model(img)
        loss = crit(pred, obj, box, cls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def infer_batch(model, images: torch.Tensor):
    model.eval()
    out = model(images)  # [N,S,S,10+C]
    b0 = out[...,:5]; b1 = out[...,5:10]; cls_logits = out[...,10:]

    # box logits → [0,1]로
    x0 = torch.sigmoid(b0[...,0]); y0 = torch.sigmoid(b0[...,1])
    w0 = torch.sigmoid(b0[...,2]); h0 = torch.sigmoid(b0[...,3]); c0 = torch.sigmoid(b0[...,4])

    x1 = torch.sigmoid(b1[...,0]); y1 = torch.sigmoid(b1[...,1])
    w1 = torch.sigmoid(b1[...,2]); h1 = torch.sigmoid(b1[...,3]); c1 = torch.sigmoid(b1[...,4])

    # class 확률
    cls_prob = F.softmax(cls_logits, dim=-1)  # [N,S,S,C]

    N = images.size(0)  # ← out.size(0)와 같지만, 입력 기준으로 맞춥니다.
    gy, gx = torch.meshgrid(
        torch.arange(S, device=out.device),
        torch.arange(S, device=out.device),
        indexing="ij"
    )
    gx = gx.float(); gy = gy.float()

    # 배치 전체 좌표로 미리 변환
    cx0 = (x0 + gx) / S; cy0 = (y0 + gy) / S
    cx1 = (x1 + gx) / S; cy1 = (y1 + gy) / S

    results = []
    for n in range(N):
        all_boxes, all_scores, all_classes = [], [], []

        for bb in range(2):  # B=2
            if bb == 0:
                cx, cy, w, h, conf = cx0[n], cy0[n], w0[n], h0[n], c0[n]  # [S,S]
            else:
                cx, cy, w, h, conf = cx1[n], cy1[n], w1[n], h1[n], c1[n]

            # 각 클래스별 점수: conf * p(class)
            scores_per_cls = conf.unsqueeze(-1) * cls_prob[n]  # [S,S,C]

            for cls_id in range(C):
                score = scores_per_cls[..., cls_id]  # [S,S]
                mask = score > CONF_THRES
                if mask.any():
                    cx_m = cx[mask]; cy_m = cy[mask]; w_m = w[mask]; h_m = h[mask]; s_m = score[mask]
                    x1m = (cx_m - w_m/2).clamp(0, 1)
                    y1m = (cy_m - h_m/2).clamp(0, 1)
                    x2m = (cx_m + w_m/2).clamp(0, 1)
                    y2m = (cy_m + h_m/2).clamp(0, 1)

                    boxes = torch.stack([x1m, y1m, x2m, y2m], dim=-1).cpu().numpy()
                    sc = s_m.detach().cpu().numpy()
                    cls_arr = np.full(sc.shape[0], cls_id, dtype=np.int32)

                    all_boxes.append(boxes); all_scores.append(sc); all_classes.append(cls_arr)

        if len(all_boxes) == 0:
            results.append((
                np.zeros((0,4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32)
            ))
        else:
            results.append((
                np.concatenate(all_boxes, 0),
                np.concatenate(all_scores, 0),
                np.concatenate(all_classes, 0)
            ))
    return results

def ap_from_pr(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))
    return ap

@torch.no_grad()
def evaluate_map(model, dataset, iou_thresholds=MAP_IOU_THRESHOLDS):
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=lambda b: list(zip(*b)))
    preds_all = []; gts_all = []
    for batch in tqdm(loader, desc="Eval/Infer", leave=False):
        imgs, objs, boxes, clss, metas = batch
        imgs = torch.stack(imgs).to(DEVICE)
        results = infer_batch(model, imgs)
        for i in range(len(results)):
            p_boxes, p_scores, p_cls = results[i]
            obj = objs[i].numpy()
            box = boxes[i].numpy()
            cls = clss[i].numpy()
            gts = []
            for gj in range(S):
                for gi in range(S):
                    if obj[gj,gi] > 0.5 and cls[gj,gi] >= 0:
                        cx = (box[gj,gi,0] + gi) / S
                        cy = (box[gj,gi,1] + gj) / S
                        w  = box[gj,gi,2]; h = box[gj,gi,3]
                        x1,y1,x2,y2 = xywh_to_xyxy_norm(cx,cy,w,h)
                        gts.append((x1,y1,x2,y2,int(cls[gj,gi])))
            preds_all.append((p_boxes, p_scores, p_cls))
            gts_all.append(gts)

    aps_50_95 = []; ap50s = []
    for iou_t in iou_thresholds:
        ap_per_class = []
        for c in range(C):
            gt_boxes_per_img = []
            total_gt = 0
            for gts in gts_all:
                gt_c = [g for g in gts if g[4] == c]
                total_gt += len(gt_c)
                gt_boxes_per_img.append(
                    np.array([[g[0], g[1], g[2], g[3]] for g in gt_c], dtype=np.float32)
                )
                if total_gt == 0:
                    continue
            dets = []  # (img_idx, score, x1,y1,x2,y2)
            for img_idx,(pb, ps, pc) in enumerate(preds_all):
                mask = (pc==c)
                if mask.sum()==0: continue
                b = pb[mask]; s = ps[mask]
                if b.shape[0]:
                    keep = nms(b.copy(), s.copy(), NMS_IOU)
                    for k in keep:
                        dets.append((img_idx, float(s[k]), *b[k].tolist()))
            if len(dets)==0:
                continue
            dets.sort(key=lambda x: x[1], reverse=True)
            tp = np.zeros(len(dets), dtype=np.float32)
            fp = np.zeros(len(dets), dtype=np.float32)
            gt_matched = [np.zeros(len(gt_boxes_per_img[i]), dtype=bool) for i in range(len(gt_boxes_per_img))]
            for d_i, det in enumerate(dets):
                img_idx, score, x1,y1,x2,y2 = det
                gt = gt_boxes_per_img[img_idx]
                if gt.size==0:
                    fp[d_i]=1; continue
                xx1 = np.maximum(x1, gt[:,0]); yy1=np.maximum(y1, gt[:,1])
                xx2 = np.minimum(x2, gt[:,2]); yy2=np.minimum(y2, gt[:,3])
                w = np.maximum(0, xx2-xx1); h=np.maximum(0, yy2-yy1)
                inter = w*h
                area_p = (x2-x1)*(y2-y1)
                area_g = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])
                ious = inter / (area_p + area_g - inter + 1e-9)
                j = int(np.argmax(ious))
                if ious[j] >= iou_t and not gt_matched[img_idx][j]:
                    tp[d_i]=1; gt_matched[img_idx][j]=True
                else:
                    fp[d_i]=1
            fp = np.cumsum(fp); tp = np.cumsum(tp)
            rec = tp / (sum(len(x) for x in gt_boxes_per_img) + 1e-9)
            prec = tp / np.maximum(tp+fp, 1e-9)
            ap = ap_from_pr(rec, prec)
            ap_per_class.append(ap)
        mAP_t = float(np.mean(ap_per_class)) if len(ap_per_class)>0 else 0.0
        if abs(iou_t-0.5)<1e-6: ap50s.append(mAP_t)
        aps_50_95.append(mAP_t)
    mAP_50 = ap50s[0] if ap50s else 0.0
    mAP_50_95 = float(np.mean(aps_50_95)) if aps_50_95 else 0.0
    return mAP_50, mAP_50_95

# ===================== 시각화 저장 =====================
def save_visuals(model, dataset, max_imgs=60):
    OUT_PRED.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    OUT_PAIR.mkdir(parents=True, exist_ok=True)
    idxs = list(range(len(dataset)))[:max_imgs]
    model.eval()
    for i in tqdm(idxs, desc="Visualize", leave=False):
        img, obj, box, cls, meta = dataset[i]
        path = meta["path"]; h0,w0 = meta["orig_shape"]
        orig = cv2.imread(path)
        with torch.no_grad():
            res = infer_batch(model, img.unsqueeze(0).to(DEVICE))[0]
        p_boxes, p_scores, p_cls = res
        gt_boxes = []; gt_labels=[]
        for gj in range(S):
            for gi in range(S):
                if obj[gj,gi] > 0.5 and cls[gj,gi] >= 0:
                    cx = (box[gj,gi,0].item() + gi) / S
                    cy = (box[gj,gi,1].item() + gj) / S
                    w  = box[gj,gi,2].item(); h = box[gj,gi,3].item()
                    x1,y1,x2,y2 = xywh_to_xyxy_norm(cx,cy,w,h)
                    gt_boxes.append([int(x1*w0), int(y1*h0), int(x2*w0), int(y2*h0)])
                    gt_labels.append(str(int(cls[gj,gi].item())))
        gt_img = draw_boxes(orig, gt_boxes, gt_labels, gt=True)

        pred_boxes = []; pred_labels=[]; pred_scores=[]
        if p_boxes.shape[0]:
            for c in range(C):
                mask = (p_cls==c)
                if mask.sum()==0: continue
                b = p_boxes[mask]; s = p_scores[mask]
                keep = nms(b.copy(), s.copy(), NMS_IOU)
                for k in keep:
                    x1,y1,x2,y2 = b[k]
                    pred_boxes.append([int(x1*w0), int(y1*h0), int(x2*w0), int(y2*h0)])
                    pred_labels.append(str(c))
                    pred_scores.append(float(s[k]))
        pred_img = draw_boxes(orig, pred_boxes, pred_labels, pred_scores, gt=False)

        bname = Path(path).stem
        p_gt   = OUT_GT / f"{bname}_gt.jpg"
        p_pred = OUT_PRED / f"{bname}_pred.jpg"
        p_pair = OUT_PAIR / f"{bname}_paired.jpg"
        cv2.imwrite(str(p_gt), gt_img)
        cv2.imwrite(str(p_pred), pred_img)
        cv2.imwrite(str(p_pair), side_by_side(gt_img, pred_img))

# ===================== 메인 =====================
def main():
    s1 = time.time()
    set_seed(SEED)
    download_coco128()

    train_ds = YoloTxtDataset(IMAGES_DIR, LABELS_DIR, augment=True)
    val_ds   = YoloTxtDataset(IMAGES_DIR, LABELS_DIR, augment=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    model = YOLOv1().to(DEVICE)
    crit = YoloV1Loss().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        loss = train_one_epoch(model, crit, train_loader, opt)
        m50, m5095 = evaluate_map(model, val_ds)
        dt = time.time()-t0
        print(f"[Epoch {ep}/{EPOCHS}] loss={loss:.4f} | mAP@50={m50:.4f} | mAP@50-95={m5095:.4f} | {dt:.1f}s")
    e1 = time.time()
    save_visuals(model, val_ds, max_imgs=60)
    print("\n✅ Done. Everything saved under:")
    print(f"   Data : {COCO128_DIR}")
    print(f"   GT   : {OUT_GT}")
    print(f"   Pred : {OUT_PRED}")
    print(f"   Pair : {OUT_PAIR}")
    print(f"   Time : {round((e1 - s1), 4)} sec")

if __name__ == "__main__":
    main()
