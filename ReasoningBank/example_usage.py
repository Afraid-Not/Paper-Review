"""
ReAct + Reflexion + ReasoningBank 사용 예시
강화학습 에피소드를 직접 정의하여 실행할 수 있습니다.
"""

from react_reflexion_reasoningbank import ReActReflexionReasoningBank
import os


def example_custom_episode():
    """사용자 정의 ReAct 시퀀스로 에피소드 실행"""
    
    # 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    bio_path = os.path.join(base_path, "miniworld_bio.csv")
    corpus_path = os.path.join(base_path, "miniworld_corpus.csv")
    world_path = os.path.join(base_path, "miniworld.csv")
    
    # 모델 생성
    model = ReActReflexionReasoningBank(
        bio_path=bio_path,
        corpus_path=corpus_path,
        world_path=world_path,
        log_dir="logs"
    )
    
    # 사용자 정의 ReAct 시퀀스
    custom_sequence = [
        # Step 1: 검색
        (
            "Liam Brook에 대한 정보를 찾기 위해 생물학 데이터베이스를 검색해야 합니다.",
            "search",
            {"query": "Liam Brook"}
        ),
        # Step 2: 추가 검색 (도시 정보)
        (
            "검색 결과에서 Seoria라는 도시가 나왔으니, 이 도시의 정보를 더 자세히 찾아보겠습니다.",
            "search",
            {"query": "Seoria"}
        ),
        # Step 3: 과거 경험 확인
        (
            "유사한 질문에 대한 과거 경험을 확인하여 더 효율적인 접근 방법을 찾겠습니다.",
            "reflect",
            {"task": "사람의 출생지 찾기"}
        ),
        # Step 4: 완료
        (
            "모든 정보를 종합하여 답변을 완성했습니다.",
            "finish",
            {"answer": "Liam Brook은 Seoria에서 태어났으며, Seoria는 Norland에 위치한 도시입니다."}
        )
    ]
    
    # 에피소드 실행
    model.run_episode(
        task_description="Liam Brook은 어디에서 태어났나요?",
        max_steps=10,
        custom_react_prompt=custom_sequence
    )


def example_multiple_episodes():
    """여러 에피소드를 연속으로 실행"""
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    bio_path = os.path.join(base_path, "miniworld_bio.csv")
    corpus_path = os.path.join(base_path, "miniworld_corpus.csv")
    world_path = os.path.join(base_path, "miniworld.csv")
    
    model = ReActReflexionReasoningBank(
        bio_path=bio_path,
        corpus_path=corpus_path,
        world_path=world_path,
        log_dir="logs"
    )
    
    # 여러 작업 정의
    tasks = [
        "Liam Brook은 어디에서 태어났나요?",
        "Ava Shin의 생년은 언제인가요?",
        "Norland의 수도는 어디인가요?",
        "Seoria는 어떤 나라에 있나요?",
        "Valenport의 인구는 얼마인가요?"
    ]
    
    print("="*60)
    print("여러 에피소드 연속 실행 시작")
    print("="*60)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"에피소드 그룹 실행 중... ({i}/{len(tasks)})")
        print(f"{'='*60}")
        model.run_episode(task, max_steps=10)
        print()


def example_with_knowledge_update():
    """지식 베이스 업데이트를 포함한 에피소드"""
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    bio_path = os.path.join(base_path, "miniworld_bio.csv")
    corpus_path = os.path.join(base_path, "miniworld_corpus.csv")
    world_path = os.path.join(base_path, "miniworld.csv")
    
    model = ReActReflexionReasoningBank(
        bio_path=bio_path,
        corpus_path=corpus_path,
        world_path=world_path,
        log_dir="logs"
    )
    
    # 지식 업데이트를 포함한 시퀀스
    custom_sequence = [
        (
            "Noah Patel에 대한 정보를 검색합니다.",
            "search",
            {"query": "Noah Patel"}
        ),
        (
            "Noah Patel의 생년 정보를 업데이트하겠습니다.",
            "update",
            {
                "key": "Noah Patel",
                "field": "born_year",
                "value": "1995",
                "source": "bio"
            }
        ),
        (
            "업데이트를 반영하고 작업을 완료합니다.",
            "finish",
            {"answer": "Noah Patel의 생년 정보를 1995로 업데이트했습니다."}
        )
    ]
    
    model.run_episode(
        task_description="Noah Patel의 생년을 1995로 업데이트하세요.",
        max_steps=10,
        custom_react_prompt=custom_sequence
    )


if __name__ == "__main__":
    print("="*60)
    print("ReAct + Reflexion + ReasoningBank 사용 예시")
    print("="*60)
    
    # 예시 1: 사용자 정의 에피소드
    print("\n[예시 1] 사용자 정의 ReAct 시퀀스")
    example_custom_episode()
    
    # 예시 2: 여러 에피소드
    print("\n[예시 2] 여러 에피소드 연속 실행")
    example_multiple_episodes()
    
    # 예시 3: 지식 업데이트
    print("\n[예시 3] 지식 베이스 업데이트 포함")
    example_with_knowledge_update()
    
    print("\n" + "="*60)
    print("모든 예시 실행 완료!")
    print("로그는 logs/ 폴더에 저장되었습니다.")
    print("각 에피소드별로 1, 2, 3, 4... 폴더가 생성되었습니다.")
    print("="*60)

