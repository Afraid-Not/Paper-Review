import torch
import numpy as np
import cv2
import os
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SAM2MultiPersonSegmentation:
    def __init__(self, checkpoint_path):
        """
        SAM2 기반 다중 인물 세그멘테이션
        
        Args:
            checkpoint_path: SAM2 체크포인트 경로
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # SAM2 Predictor 초기화 (간단한 방법)
        self.predictor = SAM2ImagePredictor.from_pretrained(
            checkpoint_path,
            device=self.device
        )
        print("SAM2 모델 로드 완료!")
    
    def segment_auto(self, image_path, num_people=5):
        """
        그리드 기반 자동 세그멘테이션 (개선 버전)
        
        Args:
            image_path: 입력 이미지 경로
            num_people: 추출할 인물 수
        
        Returns:
            masks: 개별 마스크 리스트
            image: 원본 이미지
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image)
        
        print("자동 세그멘테이션 실행 중...")
        
        # 더 촘촘한 그리드 생성 (사람 위치에 집중)
        h, w = image.shape[:2]
        
        # 사람이 있을 만한 영역에 포인트 집중
        x_points = []
        y_points = []
        
        # 상단 1/3 영역 (얼굴)
        for i in range(num_people + 2):
            x = w * (0.1 + 0.8 * i / (num_people + 1))
            y = h * 0.25  # 얼굴 높이
            x_points.append(x)
            y_points.append(y)
        
        # 중앙 영역 (몸통)
        for i in range(num_people + 2):
            x = w * (0.1 + 0.8 * i / (num_people + 1))
            y = h * 0.5  # 몸통 높이
            x_points.append(x)
            y_points.append(y)
        
        all_masks = []
        all_scores = []
        
        print(f"  {len(x_points)}개 포인트로 스캔 중...")
        
        for x, y in zip(x_points, y_points):
            input_point = np.array([[x, y]])
            input_label = np.array([1])
            
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True  # 여러 마스크 생성
                )
                
                # 점수가 높은 마스크만 선택
                for mask, score in zip(masks, scores):
                    if score > 0.85:  # 임계값 상향
                        all_masks.append(mask)
                        all_scores.append(score)
            except:
                continue
        
        print(f"  {len(all_masks)}개의 마스크 생성됨")
        
        # 중복 제거 (더 엄격하게)
        unique_masks = self._remove_duplicate_masks(all_masks, all_scores, 
                                                     iou_threshold=0.5)
        
        # 면적과 비율로 필터링 (사람 같은 것만)
        filtered_masks = []
        for mask_dict in unique_masks:
            bbox = mask_dict['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = mask_dict['area']
            
            # 사람 비율 체크 (세로가 가로보다 길어야 함)
            if height > 0 and width > 0:
                aspect_ratio = height / width
                image_area = h * w
                area_ratio = area / image_area
                
                # 사람의 일반적인 비율: 세로/가로 > 1.2, 면적 3-40%
                if (1.0 < aspect_ratio < 3.5 and 
                    0.03 < area_ratio < 0.4):
                    filtered_masks.append(mask_dict)
        
        print(f"  필터링 후 {len(filtered_masks)}개 객체")
        
        # 면적 기준 정렬 및 상위 N개 선택
        sorted_masks = sorted(filtered_masks, 
                            key=lambda x: x['area'], 
                            reverse=True)[:num_people]
        
        # 왼쪽부터 번호 매기기 (x 좌표 기준 정렬)
        sorted_masks = sorted(sorted_masks, 
                            key=lambda x: x['bbox'][0])
        
        print(f"  최종 {len(sorted_masks)}개 사람 선택됨")
        
        return sorted_masks, image
    
    def segment_with_points(self, image_path, points_list):
        """
        각 사람에 대해 포인트를 찍어서 세그멘테이션
        
        Args:
            image_path: 입력 이미지 경로
            points_list: 각 사람의 좌표 리스트 [(x1,y1), (x2,y2), ...]
        
        Returns:
            masks: 개별 마스크 리스트
            image: 원본 이미지
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(image)
        
        all_masks = []
        print(f"{len(points_list)}명에 대해 세그멘테이션 실행 중...")
        
        for idx, point in enumerate(points_list):
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])  # 1 = foreground
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # 가장 높은 점수의 마스크 선택
            best_idx = np.argmax(scores)
            all_masks.append({
                'segmentation': masks[best_idx],
                'area': int(masks[best_idx].sum()),
                'bbox': self._mask_to_bbox(masks[best_idx]),
                'point': point,
                'score': float(scores[best_idx])
            })
            print(f"  사람 {idx+1} 세그멘테이션 완료 (점수: {scores[best_idx]:.3f})")
        
        return all_masks, image
    
    def segment_interactive(self, image_path):
        """
        대화형 모드: 이미지를 띄우고 클릭으로 각 사람 선택
        
        Args:
            image_path: 입력 이미지 경로
        
        Returns:
            masks: 개별 마스크 리스트
            image: 원본 이미지
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("\n대화형 모드:")
        print("  - 좌클릭: 각 사람의 중심을 클릭")
        print("  - 우클릭: 완료")
        print("  - 창을 닫아도 완료")
        
        points = []
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title("각 사람을 클릭하세요 (우클릭 또는 창 닫기: 완료)", fontsize=14)
        ax.axis('off')
        
        def onclick(event):
            if event.button == 1 and event.xdata and event.ydata:  # 좌클릭
                x, y = int(event.xdata), int(event.ydata)
                points.append((x, y))
                ax.plot(x, y, 'r*', markersize=20, markeredgewidth=2, 
                       markeredgecolor='white')
                ax.text(x, y-20, f'{len(points)}', color='red', 
                       fontsize=14, weight='bold', ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', 
                                edgecolor='red', linewidth=2))
                plt.draw()
                print(f"  사람 {len(points)}: ({x}, {y})")
            elif event.button == 3:  # 우클릭
                plt.close()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()
        
        if len(points) == 0:
            print("포인트가 선택되지 않았습니다.")
            return [], image
        
        print(f"\n{len(points)}개 포인트 선택됨. 세그멘테이션 시작...")
        return self.segment_with_points(image_path, points)
    
    def _remove_duplicate_masks(self, masks, scores, iou_threshold=0.7):
        """중복 마스크 제거"""
        if len(masks) == 0:
            return []
        
        unique_masks = []
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            is_duplicate = False
            
            for unique_dict in unique_masks:
                unique_mask = unique_dict['segmentation']
                
                # IoU 계산
                intersection = np.logical_and(mask, unique_mask).sum()
                union = np.logical_or(mask, unique_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                    
                    if iou > iou_threshold:
                        # 더 높은 점수의 마스크로 대체
                        if score > unique_dict['score']:
                            unique_dict['segmentation'] = mask
                            unique_dict['score'] = float(score)
                            unique_dict['area'] = int(mask.sum())
                            unique_dict['bbox'] = self._mask_to_bbox(mask)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_masks.append({
                    'segmentation': mask,
                    'area': int(mask.sum()),
                    'bbox': self._mask_to_bbox(mask),
                    'score': float(score)
                })
        
        return unique_masks
    
    def _mask_to_bbox(self, mask):
        """마스크를 바운딩 박스로 변환"""
        pos = np.where(mask)
        if len(pos[0]) == 0:
            return [0, 0, 0, 0]
        ymin, ymax = int(pos[0].min()), int(pos[0].max())
        xmin, xmax = int(pos[1].min()), int(pos[1].max())
        return [xmin, ymin, xmax, ymax]
    
    def visualize_results(self, image, masks, save_path=None, show_numbers=True):
        """
        세그멘테이션 결과 시각화
        
        Args:
            image: 원본 이미지
            masks: 마스크 딕셔너리 리스트
            save_path: 저장 경로
            show_numbers: 번호 표시 여부
        """
        if len(masks) == 0:
            print("표시할 마스크가 없습니다.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 원본 이미지
        axes[0].imshow(image)
        axes[0].set_title("원본 이미지", fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # 2. 모든 마스크 오버레이
        axes[1].imshow(image)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
        
        for idx, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation'].astype(bool)  # boolean으로 변환
            color = colors[idx][:3]
            
            # 마스크 오버레이
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*color, 0.5]
            axes[1].imshow(colored_mask)
            
            # 바운딩 박스
            bbox = mask_dict['bbox']
            rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           linewidth=3, edgecolor=color, facecolor='none')
            axes[1].add_patch(rect)
            
            # 번호 표시
            if show_numbers:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                axes[1].text(cx, cy, f'{idx+1}', color='white', 
                           fontsize=24, weight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor=color, 
                                   edgecolor='white', linewidth=2))
        
        axes[1].set_title(f"전체 오버레이 ({len(masks)}명)", 
                         fontsize=14, weight='bold')
        axes[1].axis('off')
        
        # 3. 개별 마스크
        axes[2].imshow(image)
        for idx, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation'].astype(bool)  # boolean으로 변환
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = colors[idx]
            axes[2].imshow(colored_mask, alpha=0.6)
        
        axes[2].set_title("개별 마스크", fontsize=14, weight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 결과 저장: {save_path}")
        
        plt.show()
    
    def save_individual_masks(self, image, masks, output_dir="./output"):
        """
        각 사람의 마스크를 개별 이미지로 저장
        
        Args:
            image: 원본 이미지
            masks: 마스크 리스트
            output_dir: 출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation'].astype(bool)  # boolean으로 변환
            
            # 마스크 적용된 이미지
            masked_image = image.copy()
            masked_image[~mask] = [255, 255, 255]  # 배경을 흰색으로
            
            # 바운딩 박스로 크롭
            bbox = mask_dict['bbox']
            cropped = masked_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # 저장
            output_path = os.path.join(output_dir, f"person_{idx+1}.png")
            cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            print(f"  ✓ person_{idx+1}.png 저장")
        
        print(f"\n모든 이미지가 '{output_dir}' 폴더에 저장되었습니다!")


# 사용 예시
if __name__ == "__main__":

    CHECKPOINT = "facebook/sam2-hiera-large"  # Hugging Face에서 자동 다운로드
    
    # 초기화
    print("=" * 50)
    print("SAM2 다중 인물 세그멘테이션")
    print("=" * 50)
    
    segmenter = SAM2MultiPersonSegmentation(checkpoint_path=CHECKPOINT)

    print("\n" + "=" * 50)
    print("방법 3: 대화형 모드")
    print("=" * 50)
    masks, image = segmenter.segment_interactive("./refs/test/test3.jpg")
    if len(masks) > 0:
        segmenter.visualize_results(image, masks, save_path="interactive_result.png")
        segmenter.save_individual_masks(image, masks, output_dir="./output_interactive")
    
    print("\n" + "=" * 50)
    print("완료!")
    print("=" * 50)