# ReAct + Reflexion + ReasoningBank 조합별 비교

생성 일시: 2025-10-30 01:50:44

## 종합 비교표

| 조합 | 평균 성공률 | 평균 스텝 수 | 평균 검색 횟수 | 평균 Reflection 횟수 |
|------|------------|------------|--------------|-------------------|
| ReAct만 | 100.0% | 2.0 | 0.0 | 0.0 |
| Reflexion만 | 100.0% | 1.0 | 0.0 | 2.0 |
| ReasoningBank만 | 40.0% | 1.0 | 0.4 | 0.0 |
| ReAct + Reflexion | 100.0% | 2.0 | 0.0 | 1.0 |
| ReAct + ReasoningBank | 40.0% | 3.0 | 0.4 | 0.0 |
| Reflexion + ReasoningBank | 40.0% | 2.0 | 0.4 | 1.0 |
| ReAct + Reflexion + ReasoningBank | 0.0% | 0.0 | 0.0 | 0.0 |

## 질문별 상세 비교

### 질문 1: Liam Brook은 어디에서 태어났나요?

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ✅ | 2 | 0 | 0 | Liam의 출생지 정보를 찾기 위해 reasoning합니다. (ReAct만 사용) |
| Reflexion만 | ✅ | 1 | 0 | 2 | 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함 |
| ReasoningBank만 | ✅ | 1 | 1 | 0 | Liam Brook은(는) Seoria에서 태어났습니다. |
| ReAct + Reflexion | ✅ | 2 | 0 | 1 | Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인 |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | Liam Brook은(는) Seoria에서 태어났습니다. |
| Reflexion + ReasoningBank | ✅ | 2 | 1 | 1 | Reflexion 가이드 + 검색 결과: Liam Brook 관련 정보 발견 |
| ReAct + Reflexion + ReasoningBank | ❌ | 0 | 0 | 0 | 에러: 'dict' object has no attribute 'observation' |

### 질문 2: Ava Shin의 생년은 언제인가요?

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ✅ | 2 | 0 | 0 | Ava의 생년 정보를 찾기 위해 reasoning합니다. (ReAct만 사용) |
| Reflexion만 | ✅ | 1 | 0 | 2 | 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함 |
| ReasoningBank만 | ✅ | 1 | 1 | 0 | Ava Shin은(는) Valenport에서 태어났습니다. |
| ReAct + Reflexion | ✅ | 2 | 0 | 1 | Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인 |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | Ava Shin은(는) Valenport에서 태어났습니다. |
| Reflexion + ReasoningBank | ✅ | 2 | 1 | 1 | Reflexion 가이드 + 검색 결과: Ava Shin 관련 정보 발견 |
| ReAct + Reflexion + ReasoningBank | ❌ | 0 | 0 | 0 | 에러: 'dict' object has no attribute 'observation' |

### 질문 3: Norland의 수도는 어디인가요?

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ✅ | 2 | 0 | 0 | 질문을 분석하여 reasoning합니다. (ReAct만 사용) |
| Reflexion만 | ✅ | 1 | 0 | 2 | 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함 |
| ReasoningBank만 | ❌ | 1 | 0 | 0 | 검색 결과를 찾을 수 없습니다. |
| ReAct + Reflexion | ✅ | 2 | 0 | 1 | Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인 |
| ReAct + ReasoningBank | ❌ | 3 | 0 | 0 | 검색 결과를 찾을 수 없습니다. |
| Reflexion + ReasoningBank | ❌ | 2 | 0 | 1 | 검색 결과를 찾을 수 없습니다. |
| ReAct + Reflexion + ReasoningBank | ❌ | 0 | 0 | 0 | 에러: 'dict' object has no attribute 'observation' |

### 질문 4: Seoria는 어떤 나라에 있나요?

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ✅ | 2 | 0 | 0 | 질문을 분석하여 reasoning합니다. (ReAct만 사용) |
| Reflexion만 | ✅ | 1 | 0 | 2 | 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함 |
| ReasoningBank만 | ❌ | 1 | 0 | 0 | 검색 결과를 찾을 수 없습니다. |
| ReAct + Reflexion | ✅ | 2 | 0 | 1 | Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인 |
| ReAct + ReasoningBank | ❌ | 3 | 0 | 0 | 검색 결과를 찾을 수 없습니다. |
| Reflexion + ReasoningBank | ❌ | 2 | 0 | 1 | 검색 결과를 찾을 수 없습니다. |
| ReAct + Reflexion + ReasoningBank | ❌ | 0 | 0 | 0 | 에러: 'dict' object has no attribute 'observation' |

### 질문 5: Valenport의 인구는 얼마인가요?

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ✅ | 2 | 0 | 0 | 질문을 분석하여 reasoning합니다. (ReAct만 사용) |
| Reflexion만 | ✅ | 1 | 0 | 2 | 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함 |
| ReasoningBank만 | ❌ | 1 | 0 | 0 | 검색 결과를 찾을 수 없습니다. |
| ReAct + Reflexion | ✅ | 2 | 0 | 1 | Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인 |
| ReAct + ReasoningBank | ❌ | 3 | 0 | 0 | 검색 결과를 찾을 수 없습니다. |
| Reflexion + ReasoningBank | ❌ | 2 | 0 | 1 | 검색 결과를 찾을 수 없습니다. |
| ReAct + Reflexion + ReasoningBank | ❌ | 0 | 0 | 0 | 에러: 'dict' object has no attribute 'observation' |

## 조합별 상세 답변

### ReAct만

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 간단한 reasoning 수행
- **답변**: Liam의 출생지 정보를 찾기 위해 reasoning합니다. (ReAct만 사용)

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 간단한 reasoning 수행
- **답변**: Ava의 생년 정보를 찾기 위해 reasoning합니다. (ReAct만 사용)

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 간단한 reasoning 수행
- **답변**: 질문을 분석하여 reasoning합니다. (ReAct만 사용)

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 간단한 reasoning 수행
- **답변**: 질문을 분석하여 reasoning합니다. (ReAct만 사용)

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 간단한 reasoning 수행
- **답변**: 질문을 분석하여 reasoning합니다. (ReAct만 사용)

### Reflexion만

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 2
- **스텝**: 1. 과거 메모리 검색
- **답변**: 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 2
- **스텝**: 1. 과거 메모리 검색
- **답변**: 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 2
- **스텝**: 1. 과거 메모리 검색
- **답변**: 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 2
- **스텝**: 1. 과거 메모리 검색
- **답변**: 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 2
- **스텝**: 1. 과거 메모리 검색
- **답변**: 과거 경험을 바탕으로: bio 데이터베이스에서 검색해야 함

### ReasoningBank만

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 1
- **Reflection 횟수**: 0
- **스텝**: 1. 지식 베이스 검색
- **답변**: Liam Brook은(는) Seoria에서 태어났습니다.

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ✅
- **스텝 수**: 1
- **검색 횟수**: 1
- **Reflection 횟수**: 0
- **스텝**: 1. 지식 베이스 검색
- **답변**: Ava Shin은(는) Valenport에서 태어났습니다.

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ❌
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. 지식 베이스 검색
- **답변**: 검색 결과를 찾을 수 없습니다.

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ❌
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. 지식 베이스 검색
- **답변**: 검색 결과를 찾을 수 없습니다.

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ❌
- **스텝 수**: 1
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. 지식 베이스 검색
- **답변**: 검색 결과를 찾을 수 없습니다.

### ReAct + Reflexion

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Question 분석 및 Reasoning, 2. Reflexion 메모리 확인
- **답변**: Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Question 분석 및 Reasoning, 2. Reflexion 메모리 확인
- **답변**: Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Question 분석 및 Reasoning, 2. Reflexion 메모리 확인
- **답변**: Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Question 분석 및 Reasoning, 2. Reflexion 메모리 확인
- **답변**: Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Question 분석 및 Reasoning, 2. Reflexion 메모리 확인
- **답변**: Reasoning + Reflection: 이름으로 검색 후 bio 정보 확인

### ReAct + ReasoningBank

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ✅
- **스텝 수**: 3
- **검색 횟수**: 1
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 지식 베이스 검색, 3. 검색 결과 분석
- **답변**: Liam Brook은(는) Seoria에서 태어났습니다.

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ✅
- **스텝 수**: 3
- **검색 횟수**: 1
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 지식 베이스 검색, 3. 검색 결과 분석
- **답변**: Ava Shin은(는) Valenport에서 태어났습니다.

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ❌
- **스텝 수**: 3
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 지식 베이스 검색, 3. 검색 결과 분석
- **답변**: 검색 결과를 찾을 수 없습니다.

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ❌
- **스텝 수**: 3
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 지식 베이스 검색, 3. 검색 결과 분석
- **답변**: 검색 결과를 찾을 수 없습니다.

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ❌
- **스텝 수**: 3
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 1. Question 분석, 2. 지식 베이스 검색, 3. 검색 결과 분석
- **답변**: 검색 결과를 찾을 수 없습니다.

### Reflexion + ReasoningBank

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 1
- **Reflection 횟수**: 1
- **스텝**: 1. Reflexion 메모리 확인, 2. 지식 베이스 검색
- **답변**: Reflexion 가이드 + 검색 결과: Liam Brook 관련 정보 발견

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ✅
- **스텝 수**: 2
- **검색 횟수**: 1
- **Reflection 횟수**: 1
- **스텝**: 1. Reflexion 메모리 확인, 2. 지식 베이스 검색
- **답변**: Reflexion 가이드 + 검색 결과: Ava Shin 관련 정보 발견

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ❌
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Reflexion 메모리 확인, 2. 지식 베이스 검색
- **답변**: 검색 결과를 찾을 수 없습니다.

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ❌
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Reflexion 메모리 확인, 2. 지식 베이스 검색
- **답변**: 검색 결과를 찾을 수 없습니다.

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ❌
- **스텝 수**: 2
- **검색 횟수**: 0
- **Reflection 횟수**: 1
- **스텝**: 1. Reflexion 메모리 확인, 2. 지식 베이스 검색
- **답변**: 검색 결과를 찾을 수 없습니다.

### ReAct + Reflexion + ReasoningBank

#### 질문 1: Liam Brook은 어디에서 태어났나요?
- **성공**: ❌
- **스텝 수**: 0
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 
- **답변**: 에러: 'dict' object has no attribute 'observation'

#### 질문 2: Ava Shin의 생년은 언제인가요?
- **성공**: ❌
- **스텝 수**: 0
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 
- **답변**: 에러: 'dict' object has no attribute 'observation'

#### 질문 3: Norland의 수도는 어디인가요?
- **성공**: ❌
- **스텝 수**: 0
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 
- **답변**: 에러: 'dict' object has no attribute 'observation'

#### 질문 4: Seoria는 어떤 나라에 있나요?
- **성공**: ❌
- **스텝 수**: 0
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 
- **답변**: 에러: 'dict' object has no attribute 'observation'

#### 질문 5: Valenport의 인구는 얼마인가요?
- **성공**: ❌
- **스텝 수**: 0
- **검색 횟수**: 0
- **Reflection 횟수**: 0
- **스텝**: 
- **답변**: 에러: 'dict' object has no attribute 'observation'

