# fix_orientation.py  (VS Code에서 F5로 실행)
import os, sys
from PIL import Image, ImageOps

# 수정할 폴더 경로만 바꿔 주세요
INPUT_DIR = r"D:\D_Study\refs"   # 이미지가 있는 폴더
OVERWRITE = True                 # True: 덮어쓰기, False: _fixed 폴더에 저장

SUPPORTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def out_path(p):
    if OVERWRITE: return p
    base = os.path.abspath(INPUT_DIR)
    rel = os.path.relpath(p, base)
    dst = os.path.join(base + "_fixed", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return dst

cnt = 0
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith(SUPPORTS):
            src = os.path.join(root, f)
            try:
                img = Image.open(src)
                # EXIF 회전 적용 → 실제 픽셀 회전
                img = ImageOps.exif_transpose(img)
                # EXIF 제거(회전 태그 무력화)
                img.save(out_path(src), exif=b"")
                cnt += 1
            except Exception as e:
                print("skip:", src, "-", e)
print(f"fixed {cnt} image(s)")
