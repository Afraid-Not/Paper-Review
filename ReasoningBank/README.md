# ReAct + Reflexion + ReasoningBank 통합 강화학습 모델

ReAct (Reasoning and Acting), Reflexion, 그리고 ReasoningBank를 결합한 강화학습 모델입니다.

## 주요 기능

1. **ReAct**: Thought-Action-Observation 패턴을 통한 추론 및 행동
2. **Reflexion**: 이전 에피소드에서 학습한 메모리 저장 및 활용
3. **ReasoningBank**: 지식 베이스 저장, 검색, 업데이트
4. **상세 로깅**: 모든 reasoning, memory, bank memory를 회차별 폴더에 저장

## 구조

```
ReasoningBank/
├── react_reflexion_reasoningbank.py  # 메인 모델 코드
├── example_usage.py                  # 사용 예시
├── miniworld_bio.csv                  # 생물 정보 데이터
├── miniworld_corpus.csv               # 도시 정보 데이터
├── miniworld.csv                      # 국가 정보 데이터
└── logs/                              # 로그 저장 폴더 (자동 생성)
    ├── 1/                             # 에피소드 1
    │   ├── react_reasoning.json       # ReAct reasoning 로그
    │   ├── reflexion_memory.json      # Reflexion 메모리 로그
    │   ├── reasoning_bank_memory.json # ReasoningBank 메모리 로그
    │   └── episode_summary.json       # 에피소드 요약
    ├── 2/                             # 에피소드 2
    │   └── ...
    └── ...
```

## 설치

```bash
pip install pandas
```

## 기본 사용법

### 1. 자동 ReAct 시퀀스 사용

```python
from react_reflexion_reasoningbank import ReActReflexionReasoningBank
import os

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

# 에피소드 실행 (자동 ReAct 시퀀스 생성)
model.run_episode("Liam Brook은 어디에서 태어났나요?", max_steps=10)
```

### 2. 사용자 정의 ReAct 시퀀스 사용

```python
# 사용자 정의 ReAct 시퀀스
custom_sequence = [
    # (thought, action, action_input)
    (
        "정보를 검색해야 합니다.",
        "search",
        {"query": "Liam Brook"}
    ),
    (
        "과거 경험을 확인합니다.",
        "reflect",
        {"task": "사람의 출생지 찾기"}
    ),
    (
        "작업을 완료합니다.",
        "finish",
        {"answer": "Liam Brook은 Seoria에서 태어났습니다."}
    )
]

model.run_episode(
    task_description="Liam Brook은 어디에서 태어났나요?",
    max_steps=10,
    custom_react_prompt=custom_sequence
)
```

### 3. 여러 에피소드 연속 실행

```python
tasks = [
    "Liam Brook은 어디에서 태어났나요?",
    "Ava Shin의 생년은 언제인가요?",
    "Norland의 수도는 어디인가요?"
]

for task in tasks:
    model.run_episode(task, max_steps=10)
```

## 지원하는 액션

### 1. `search`
지식 베이스에서 정보 검색

```python
{
    "action": "search",
    "action_input": {"query": "Liam Brook"}
}
```

### 2. `update`
지식 베이스 업데이트

```python
{
    "action": "update",
    "action_input": {
        "key": "Liam Brook",
        "field": "born_year",
        "value": "1992",
        "source": "bio"  # "bio", "corpus", "world"
    }
}
```

### 3. `reflect`
과거 경험 확인 (Reflexion)

```python
{
    "action": "reflect",
    "action_input": {"task": "사람의 출생지 찾기"}
}
```

### 4. `finish`
작업 완료

```python
{
    "action": "finish",
    "action_input": {"answer": "답변 내용"}
}
```

## 로그 구조

각 에피소드마다 `logs/{episode_num}/` 폴더에 다음 파일들이 저장됩니다:

### 1. `react_reasoning.json`
ReAct의 reasoning 과정 전체 기록

```json
{
  "total_steps": 3,
  "steps": [
    {
      "step_num": 1,
      "thought": "...",
      "action": "search",
      "action_input": {...},
      "observation": "...",
      "timestamp": "..."
    },
    ...
  ]
}
```

### 2. `reflexion_memory.json`
Reflexion의 모든 메모리 기록

```json
{
  "total_memories": 5,
  "memories": [
    {
      "episode_num": 1,
      "task_description": "...",
      "success": true,
      "failure_reason": null,
      "lesson_learned": "...",
      "timestamp": "..."
    },
    ...
  ]
}
```

### 3. `reasoning_bank_memory.json`
ReasoningBank의 검색 및 업데이트 히스토리

```json
{
  "total_memories": 10,
  "memories": [
    {
      "query": "Liam Brook",
      "retrieved_knowledge": [...],
      "updated_knowledge": null,
      "timestamp": "..."
    },
    ...
  ]
}
```

### 4. `episode_summary.json`
에피소드 요약

```json
{
  "episode_num": 1,
  "task": "...",
  "success": true,
  "total_steps": 3,
  "timestamp": "..."
}
```

## 실행 예시

```bash
# 기본 실행
python react_reflexion_reasoningbank.py

# 사용 예시 실행
python example_usage.py
```

## 특징

1. **회차별 로그 저장**: 각 에피소드마다 별도 폴더에 모든 로그 저장
2. **완전한 추적**: ReAct reasoning, Reflexion memory, ReasoningBank memory 모두 기록
3. **유연한 시퀀스 정의**: 자동 생성 또는 사용자 정의 시퀀스 사용 가능
4. **강화학습 지원**: 이전 에피소드 학습을 통한 점진적 개선

## 데이터 형식

### miniworld_bio.csv
사람의 생물 정보

| title | type | born_in_city | born_year | bio |
|-------|------|--------------|-----------|-----|
| Liam Brook | person | Seoria | 1991 | ... |

### miniworld_corpus.csv
도시 정보

| title | type | country | summary | population_k |
|-------|------|---------|---------|--------------|
| Seoria | city | Norland | ... | 209 |

### miniworld.csv
국가 정보

| title | type | capital | currency |
|-------|------|---------|----------|
| Norland | country | Hadria | NOR |

## 주의사항

- 각 에피소드는 순차적으로 실행되며, 이전 에피소드의 Reflexion 메모리가 다음 에피소드에 영향을 줍니다.
- 지식 베이스 업데이트는 현재 세션에만 유효하며, CSV 파일 자체는 수정하지 않습니다.
- 로그 폴더는 자동으로 생성되며, 기존 폴더가 있으면 덮어씁니다.

