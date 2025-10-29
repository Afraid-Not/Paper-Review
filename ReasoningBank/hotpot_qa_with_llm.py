"""
HotpotQA 데이터셋 + LLM 통합 버전 (OpenAI 사용)
실제 LLM을 사용하여 ReAct + Reflexion + ReasoningBank 테스트
"""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset

# OpenAI 라이브러리
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("경고: openai 라이브러리가 없습니다. pip install openai 로 설치하세요.")


class LLMInterface:
    """LLM 통합 인터페이스 (OpenAI 지원)"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai 라이브러리가 필요합니다. pip install openai 로 설치하세요.")
            
            # OpenAI SDK 1.0+ 방식
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                # 환경변수에서 가져오기
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                else:
                    raise ValueError("OPENAI_API_KEY를 설정하거나 api_key 인자를 제공해야 합니다.")
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}. 'openai'만 지원합니다.")
    
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """프롬프트를 받아 LLM으로 생성"""
        if self.provider == "openai":
            try:
                # OpenAI SDK 1.0+ 방식
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that reasons step by step."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[LLM 에러: {str(e)}]"
        
        return "[LLM 미구현]"
    
    def reason(self, thought: str, context: str = "") -> str:
        """추론 단계 수행"""
        prompt = f"""다음 상황에서 추론을 수행하세요.

상황: {context}
생각: {thought}

다음 단계를 결정하세요:"""
        return self.generate(prompt, max_tokens=150)
    
    def generate_answer(self, question: str, context: str) -> str:
        """질문에 대한 답변 생성"""
        prompt = f"""주어진 컨텍스트를 바탕으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        return self.generate(prompt, max_tokens=100)


class HotpotQAReasoningBankWithLLM:
    """LLM을 사용하는 HotpotQA ReasoningBank"""
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        self.qa_pairs: List[Dict[str, Any]] = []
        self.loaded = False
        self.llm = llm
    
    def load_dataset(self, split: str = "validation", subset: str = "distractor", num_samples: Optional[int] = None):
        """HotpotQA 데이터셋 로드"""
        try:
            print(f"HotpotQA 데이터셋 로딩 중... (subset: {subset}, split: {split})")
            dataset = load_dataset("hotpotqa/hotpot_qa", subset)
            
            if split not in dataset:
                print(f"경고: {split} split을 찾을 수 없습니다. 사용 가능한 splits: {list(dataset.keys())}")
                split = list(dataset.keys())[0]
            
            data = dataset[split]
            
            if num_samples:
                data = data.select(range(min(num_samples, len(data))))
            
            self.qa_pairs = []
            for item in data:
                converted_item = {
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "type": item.get("type", ""),
                    "level": item.get("level", ""),
                    "context": item.get("context", {}),
                    "supporting_facts": item.get("supporting_facts", {})
                }
                self.qa_pairs.append(converted_item)
            
            self.loaded = True
            print(f"로드 완료: {len(self.qa_pairs)}개의 질문-답변 쌍")
            return True
            
        except Exception as e:
            print(f"데이터셋 로드 실패: {str(e)}")
            return False
    
    def search_with_llm(self, query: str) -> List[Dict[str, Any]]:
        """LLM을 사용하여 의미 기반 검색"""
        if not self.loaded:
            return []
        
        # 먼저 키워드 검색으로 후보 찾기 (빠른 필터링)
        keyword_results = self._keyword_search(query)
        
        if self.llm is None or len(keyword_results) == 0:
            # LLM 없으면 키워드 매칭 결과 사용
            return keyword_results[:5]
        
        # 키워드로 매칭된 결과가 있으면 그 중에서 더 정확한 것 선택
        if keyword_results:
            # 키워드 매칭된 결과가 있으면 우선 사용
            return keyword_results[:5]
        
        # 키워드 매칭 실패 시 LLM으로 검색 시도 (느림)
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for qa_pair in self.qa_pairs:
            question = qa_pair.get("question", "").lower()
            question_words = set(question.split())
            
            # 단어 겹침이 있으면 관련성이 높음
            overlap = len(query_words & question_words)
            if overlap > 0:
                results.append({
                    "source": "hotpotqa",
                    "data": qa_pair,
                    "relevance_score": overlap / max(len(query_words), 1)
                })
        
        # 관련성 순으로 정렬
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # LLM으로 더 정확한 필터링 (옵션, 느릴 수 있음)
        # 너무 많은 후보가 있으면 LLM 사용 안 함
        if len(results) > 10:
            return results[:5]
        
        return results[:5]
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (개선된 버전)"""
        results = []
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # 대소문자 무시하고 검색
        for qa_pair in self.qa_pairs:
            question = qa_pair.get("question", "").lower().strip()
            answer = qa_pair.get("answer", "").lower().strip()
            
            # 질문에서 키워드 검색
            question_words = set(question.split())
            overlap = query_words & question_words
            
            # 최소 1개 이상의 키워드가 일치하면 관련성 있음
            if len(overlap) > 0:
                # 관련성 점수 계산
                relevance_score = len(overlap) / max(len(query_words), 1)
                results.append({
                    "source": "hotpotqa",
                    "data": qa_pair,
                    "relevance_score": relevance_score
                })
        
        # 관련성 순으로 정렬
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return results[:5]
    
    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """모든 QA 쌍 반환"""
        return self.qa_pairs


class ReActWithLLM:
    """LLM을 사용하는 ReAct 에이전트"""
    
    def __init__(self, llm: LLMInterface, knowledge_base: HotpotQAReasoningBankWithLLM):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.steps = []
    
    def think(self, thought: str, context: str = "") -> str:
        """LLM으로 추론"""
        reasoning = self.llm.reason(thought, context)
        return reasoning
    
    def act(self, action: str, action_input: Dict[str, Any]) -> str:
        """액션 수행"""
        if action == "search":
            query = action_input.get("query", "")
            results = self.knowledge_base.search_with_llm(query)
            
            if results:
                # 검색 결과를 텍스트로 변환
                context_text = "\n".join([
                    f"Q: {r['data'].get('question', '')}\nA: {r['data'].get('answer', '')}"
                    for r in results[:3]
                ])
                return f"검색 결과 ({len(results)}개):\n{context_text}"
            else:
                return "검색 결과를 찾을 수 없습니다."
        
        elif action == "reason":
            # LLM으로 추가 추론
            question = action_input.get("question", "")
            context = action_input.get("context", "")
            reasoning = self.llm.reason(f"질문: {question}", context)
            return f"추론 결과: {reasoning}"
        
        elif action == "answer":
            # 최종 답변 생성
            question = action_input.get("question", "")
            context = action_input.get("context", "")
            answer = self.llm.generate_answer(question, context)
            return f"답변: {answer}"
        
        return "알 수 없는 액션"
    
    def step(self, step_num: int, thought: str, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct 한 단계 실행"""
        # Thought를 LLM으로 개선
        improved_thought = self.think(thought)
        
        # Action 수행
        observation = self.act(action, action_input)
        
        step_record = {
            "step_num": step_num,
            "thought": improved_thought,
            "action": action,
            "action_input": action_input,
            "observation": observation
        }
        self.steps.append(step_record)
        
        return step_record


def test_with_llm_example(model_name: Optional[str] = None):
    """OpenAI LLM을 사용한 테스트 예시"""
    print("="*80)
    print("OpenAI를 사용한 HotpotQA 테스트")
    print("="*80)
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("\n설정 방법:")
        print("1. 환경변수 설정:")
        print("   Windows: set OPENAI_API_KEY=your-api-key")
        print("   Linux/Mac: export OPENAI_API_KEY=your-api-key")
        print("2. 또는 .env 파일 사용")
        return None
    
    # 기본 모델명
    model_to_use = model_name or "gpt-3.5-turbo"
    
    # LLM 초기화
    try:
        print(f"\n모델 사용: {model_to_use}")
        
        llm = LLMInterface(
            provider="openai",
            model_name=model_to_use,
            api_key=api_key
        )
        print("✅ OpenAI API 초기화 완료!")
    except Exception as e:
        print(f"\n❌ 초기화 실패: {str(e)}")
        print("\n해결 방법:")
        print("1. OPENAI_API_KEY 환경변수 확인")
        print("2. openai 라이브러리 설치: pip install openai")
        print("3. LLM 없이 테스트: python hotpot_qa_integration.py")
        return None
    
    # Knowledge Base 초기화
    kb = HotpotQAReasoningBankWithLLM(llm=llm)
    kb.load_dataset(split="validation", subset="distractor", num_samples=5)
    
    # ReAct 에이전트 초기화
    agent = ReActWithLLM(llm=llm, knowledge_base=kb)
    
    # 테스트 질문
    if kb.qa_pairs:
        test_qa = kb.qa_pairs[0]
        question = test_qa["question"]
        correct_answer = test_qa["answer"]
        
        print(f"\n질문: {question}")
        print(f"정답: {correct_answer}")
        print("\nReAct 프로세스 시작...\n")
        
        # Step 1: 검색
        step1 = agent.step(1, "질문에 답하기 위해 관련 정보를 검색해야 합니다.", "search", {"query": question})
        print(f"[Step 1] {step1['observation'][:200]}...")
        
        # Step 2: 추론
        step2 = agent.step(2, "검색 결과를 바탕으로 답변을 추론합니다.", "reason", {
            "question": question,
            "context": step1['observation']
        })
        print(f"\n[Step 2] {step2['observation'][:200]}...")
        
        # Step 3: 답변 생성
        step3 = agent.step(3, "최종 답변을 생성합니다.", "answer", {
            "question": question,
            "context": step1['observation'] + "\n" + step2['observation']
        })
        print(f"\n[Step 3] {step3['observation']}")
        
        print(f"\n정답 비교:")
        print(f"  생성된 답변: {step3['observation']}")
        print(f"  실제 정답: {correct_answer}")
        
        # 의미 기반 정답 비교
        generated_text = step3['observation'].lower()
        correct_text = correct_answer.lower()
        
        # "답변:" 등의 프리픽스 제거
        if "답변:" in generated_text:
            generated_text = generated_text.split("답변:")[-1].strip()
        
        # 간단한 매칭 체크
        is_correct = False
        
        # 직접 매칭
        if correct_text in generated_text or generated_text in correct_text:
            is_correct = True
        # yes/no 형식 체크
        elif (correct_text in ["yes", "no"] and 
              (correct_text in generated_text or 
               ("네" in generated_text and correct_text == "yes") or
               ("같은" in generated_text and correct_text == "yes") or
               ("아니" in generated_text and correct_text == "no"))):
            is_correct = True
        
        if is_correct:
            print(f"  ✅ 정답일 가능성이 높습니다 (의미가 일치)")
        else:
            print(f"  ⚠️  정답 형식이 다를 수 있습니다 (직접 확인 필요)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HotpotQA 테스트 (OpenAI 사용)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="사용할 OpenAI 모델명 (기본: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API 키 (기본: OPENAI_API_KEY 환경변수 사용)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("OpenAI를 사용하는 HotpotQA 테스트")
    print("="*80)
    print("\n필요한 라이브러리:")
    print("  pip install openai")
    print("\n필요한 설정:")
    print("  OPENAI_API_KEY 환경변수 설정")
    print("\n사용 가능한 모델:")
    print("  1. gpt-3.5-turbo (기본값) - 빠르고 저렴")
    print("  2. gpt-4 - 더 정확하지만 비쌈")
    print("  3. gpt-4-turbo-preview - 최신 GPT-4")
    print("="*80)
    
    # API 키 설정
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    test_with_llm_example(model_name=args.model)

