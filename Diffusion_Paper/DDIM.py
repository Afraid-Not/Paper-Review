import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def setup_model():
    """
    사전 학습된 InstructPix2Pix 모델 로드 (DDIM 스케줄러 사용)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model_id = "timbrooks/instruct-pix2pix"
    
    print("모델 로딩 중... (처음 실행 시 다운로드에 시간이 걸릴 수 있습니다)")
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        use_safetensors=True
    )
    
    pipe = pipe.to(device)
    
    # DDIM 스케줄러
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print("모델 로딩 완료! (DDIM 스케줄러 사용)")
    return pipe, device

def age_transform_single(image_path, pipe, device):
    """
    하나의 얼굴 이미지를 늙게 변환합니다
    
    Args:
        image_path: 입력 이미지 경로
        pipe: Diffusion 파이프라인
        device: 디바이스 (cuda/cpu)
    
    Returns:
        original: 원본 이미지
        aged: 늙은 버전 이미지
    """
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 크기 조정 (512x512가 최적)
    image = image.resize((512, 512))
    
    # 노화 프롬프트
    prompt = "make the person look like an elderly grandfather, aged 70-80 years old, with wrinkles and gray hair"
    
    print(f"\n'{image_path}' 변환 중...")
    print(f"프롬프트: {prompt}")
    
    # 이미지 변환
    aged = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=50,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
    ).images[0]
    
    return image, aged

def process_two_images(image_path1, image_path2, pipe, device):
    """
    두 개의 이미지를 각각 늙게 변환합니다
    
    Args:
        image_path1: 첫 번째 이미지 경로
        image_path2: 두 번째 이미지 경로
        pipe: Diffusion 파이프라인
        device: 디바이스
    """
    print("\n" + "="*60)
    print("두 개 사진 노화 변환 시작")
    print("="*60)
    
    # 첫 번째 이미지 변환
    print("\n[1/2] 첫 번째 사진 처리 중...")
    original1, aged1 = age_transform_single(image_path1, pipe, device)
    
    # 두 번째 이미지 변환
    print("\n[2/2] 두 번째 사진 처리 중...")
    original2, aged2 = age_transform_single(image_path2, pipe, device)
    
    # 결과 시각화
    visualize_two_results(original1, aged1, original2, aged2, 
                          image_path1, image_path2)
    
    return (original1, aged1), (original2, aged2)

def visualize_two_results(original1, aged1, original2, aged2, 
                          path1, path2):
    """
    두 개의 원본과 변환된 이미지들을 시각화합니다
    """
    plt.figure(figsize=(16, 8))
    
    # 첫 번째 사진 세트
    plt.subplot(2, 2, 1)
    plt.imshow(original1)
    plt.title(f"Original 1\n({path1.split('/')[-1]})", 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(aged1)
    plt.title("Aged Version 1\n(70-80 years old)", 
              fontsize=12, fontweight='bold', color='darkred')
    plt.axis('off')
    
    # 두 번째 사진 세트
    plt.subplot(2, 2, 3)
    plt.imshow(original2)
    plt.title(f"Original 2\n({path2.split('/')[-1]})", 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(aged2)
    plt.title("Aged Version 2\n(70-80 years old)", 
              fontsize=12, fontweight='bold', color='darkred')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('two_faces_aging_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ 결과가 'two_faces_aging_results.png'로 저장되었습니다.")

def save_individual_results(original1, aged1, original2, aged2):
    """
    각 결과를 개별 파일로도 저장합니다
    """
    aged1.save('aged_photo1.png')
    aged2.save('aged_photo2.png')
    
    print(f"✅ 개별 파일도 저장 완료:")
    print(f"   - aged_photo1.png")
    print(f"   - aged_photo2.png")

if __name__ == '__main__':
    # 모델 설정
    pipe, device = setup_model()
    
    print("\n" + "="*60)
    print("얼굴 나이 변환 Diffusion 모델 (2장 동시 처리)")
    print("="*60)
    
    # 🔥 여기에 두 개의 이미지 경로를 입력하세요! 🔥
    image_path1 = "./refs/test/test3.jpg"  # <- 첫 번째 사진 경로
    image_path2 = "./refs/test/test4.jpg"  # <- 두 번째 사진 경로
    
    # 두 이미지 처리
    (orig1, aged1), (orig2, aged2) = process_two_images(
        image_path1, image_path2, pipe, device
    )
    
    # 개별 파일로도 저장
    save_individual_results(orig1, aged1, orig2, aged2)
    
    print("\n" + "="*60)
    print("✅ 모든 작업 완료!")
    print("="*60)
    print(f"CUDA 사용: {torch.cuda.is_available()}")
    print(f"스케줄러: DDIM (Deterministic Sampling)")
    print("\n💡 TIP:")
    print("- 더 강한 노화 효과: guidance_scale 값을 높이세요 (7.5 → 10)")
    print("- 더 높은 품질: num_inference_steps를 늘리세요 (50 → 100)")
    print("- 원본 유사도 조절: image_guidance_scale 조정 (1.5 → 1.0~2.0)")