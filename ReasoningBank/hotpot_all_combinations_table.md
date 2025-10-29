# HotpotQA + OpenAI를 사용한 모든 조합 비교

생성 일시: 2025-10-30 02:34:43

## 종합 비교표

| 조합 | 평균 성공률 | 평균 스텝 수 | 평균 검색 횟수 | 평균 Reflection 횟수 |
|------|------------|------------|--------------|-------------------|
| ReAct만 | 20.0% | 2.0 | 0.0 | 0.0 |
| Reflexion만 | 0.0% | 1.0 | 0.0 | 2.4 |
| ReasoningBank만 | 100.0% | 1.0 | 3.4 | 0.0 |
| ReAct + Reflexion | 40.0% | 2.0 | 0.0 | 2.0 |
| ReAct + ReasoningBank | 100.0% | 3.0 | 1.0 | 0.0 |
| Reflexion + ReasoningBank | 80.0% | 2.0 | 3.4 | 2.0 |
| ReAct + Reflexion + ReasoningBank | 100.0% | 3.0 | 1.0 | 2.0 |

## 질문별 상세 비교

### 질문 1: Were Scott Derrickson and Ed Wood of the same nationality?...

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ❌ | 2 | 0 | 0 | 답변: Scott Derrickson and Ed Wood both being Americ... |
| Reflexion만 | ❌ | 1 | 0 | 1 | Scott Derrickson and Ed Wood were not of the same ... |
| ReasoningBank만 | ✅ | 1 | 1 | 0 | Yes, Scott Derrickson and Ed Wood were of the same... |
| ReAct + Reflexion | ✅ | 2 | 0 | 2 | 답변: 네, Scott Derrickson과 Ed Wood는 동일한 국적인 미국인입니다. ... |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | 답변: 예, Scott Derrickson과 Ed Wood는 같은 국적이었습니다. |
| Reflexion + ReasoningBank | ❌ | 2 | 1 | 2 | 검색 결과에 따르면, Scott Derrickson과 Ed Wood는 동일한 국적이었습니다... |
| ReAct + Reflexion + ReasoningBank | ✅ | 3 | 1 | 2 | 답변: 네, Scott Derrickson과 Ed Wood는 동일한 국적을 가지고 있습니다... |

### 질문 2: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tel...

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ❌ | 2 | 0 | 0 | 답변: 주어진 질문에 대한 답변을 찾기 위해 "Kiss and Tell" 영화에서 Corl... |
| Reflexion만 | ❌ | 1 | 0 | 2 | 주어진 질문은 영화 'Kiss and Tell'에서 Corliss Archer 역할을 한 ... |
| ReasoningBank만 | ✅ | 1 | 3 | 0 | The government position held by the woman who port... |
| ReAct + Reflexion | ❌ | 2 | 0 | 2 | 답변: 1. 먼저, 영화 "Kiss and Tell"에 나온 여성 주연 배우의 이름을 찾아... |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | 답변: The government position held by the woman who ... |
| Reflexion + ReasoningBank | ✅ | 2 | 3 | 2 | The government position held by the woman who port... |
| ReAct + Reflexion + ReasoningBank | ✅ | 3 | 1 | 2 | 답변: The woman who portrayed Corliss Archer in the ... |

### 질문 3: What science fantasy young adult series, told in first person, has a set of companion books narratin...

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ❌ | 2 | 0 | 0 | 답변: 주어진 정보에 따르면, 과학 판타지 장르의 청소년을 대상으로 한 첫 인칭으로 이야기... |
| Reflexion만 | ❌ | 1 | 0 | 3 | 주어진 정보로는 정확히 어떤 시리즈인지 알기 어려운 것 같습니다. 더 자세한 정보가 필요할... |
| ReasoningBank만 | ✅ | 1 | 3 | 0 | Animorphs |
| ReAct + Reflexion | ❌ | 2 | 0 | 2 | 답변: 주어진 정보에 따르면, 사용자가 찾는 책은 과학 판타지 장르의 청소년 시리즈이며, ... |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | 답변: The science fantasy young adult series, told i... |
| Reflexion + ReasoningBank | ✅ | 2 | 3 | 2 | 주어진 정보에 따르면, 해당 소설은 "Animorphs"이라고 추정됩니다. |
| ReAct + Reflexion + ReasoningBank | ✅ | 3 | 1 | 2 | 답변: The answer is "Animorphs." |

### 질문 4: Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?...

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ✅ | 2 | 0 | 0 | 답변: No, the Laleli Mosque and Esma Sultan Mansion ... |
| Reflexion만 | ❌ | 1 | 0 | 3 | 주어진 컨텍스트만으로는 Laleli Mosque와 Esma Sultan Mansion이 같... |
| ReasoningBank만 | ✅ | 1 | 5 | 0 | 아니요. |
| ReAct + Reflexion | ✅ | 2 | 0 | 2 | 답변: Given the inference that the two buildings are... |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | 답변: Based on the search results provided, it was c... |
| Reflexion + ReasoningBank | ✅ | 2 | 5 | 2 | 컨텍스트를 고려하면, Laleli Mosque와 Esma Sultan Mansion이 동일... |
| ReAct + Reflexion + ReasoningBank | ✅ | 3 | 1 | 2 | 답변: No, according to the search results, the Lalel... |

### 질문 5: The director of the romantic comedy "Big Stone Gap" is based in what New York city?...

| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |
|------|------|---------|------|------------|----------|
| ReAct만 | ❌ | 2 | 0 | 0 | 답변: 1. "Big Stone Gap"의 감독을 확인해야 합니다.
2. "Big Ston... |
| Reflexion만 | ❌ | 1 | 0 | 3 | 질문은 영화 "Big Stone Gap"의 로맨틱 코미디 감독이 뉴욕 시 어디에 기반을 두... |
| ReasoningBank만 | ✅ | 1 | 5 | 0 | The director of the romantic comedy "Big Stone Gap... |
| ReAct + Reflexion | ❌ | 2 | 0 | 2 | 답변: 1. 먼저, 영화 "Big Stone Gap"의 감독을 확인해야 합니다.
2. "B... |
| ReAct + ReasoningBank | ✅ | 3 | 1 | 0 | 답변: The director of the romantic comedy "Big Stone... |
| Reflexion + ReasoningBank | ✅ | 2 | 5 | 2 | The director of the romantic comedy "Big Stone Gap... |
| ReAct + Reflexion + ReasoningBank | ✅ | 3 | 1 | 2 | 답변: The director of the romantic comedy "Big Stone... |

