"""
HotpotQA 데이터셋 + LLM 통합 버전 (Llama 사용)
실제 LLM을 사용하여 ReAct + Reflexion + ReasoningBank 테스트
"""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset

# Llama 모델 라이브러리
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("경고: transformers 라이브러리가 없습니다. pip install transformers torch 로 설치하세요.")

# OpenAI (옵션, 폴백용)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMInterface:
    """LLM 통합 인터페이스 (Llama 모델 지원)"""
    
    def __init__(self, 
                 provider: str = "llama",
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",  # HuggingFace 모델명
                 device: str = "auto",
                 load_in_8bit: bool = False):
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        if provider == "llama" or provider == "local":
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers 라이브러리가 필요합니다. pip install transformers torch")
            
            self._load_llama_model()
        elif provider == "openai":
            if HAS_OPENAI:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    openai.api_key = api_key
                else:
                    print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
            else:
                raise ImportError("openai 라이브러리가 필요합니다.")
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}")
    
    def _load_llama_model(self):
        """Llama 모델 로드"""
        print(f"Llama 모델 로딩 중: {self.model_name}")
        print("처음 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다...")
        
        try:
            # device 자동 설정
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"사용 중인 디바이스: {self.device}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 모델 로드 옵션
            model_kwargs = {}
            if self.load_in_8bit and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except:
                    print("경고: 8-bit 양자화를 사용할 수 없습니다. 전체 모델을 로드합니다.")
            
            # 모델 로드
            use_device_map = self.device == "cuda"
            
            if use_device_map:
                # CUDA 사용 시 device_map="auto"로 설정 (accelerate 사용)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    **model_kwargs
                )
                # device_map을 사용하면 모델이 이미 적절한 디바이스에 배치됨
                pipeline_device = None  # device_map 사용 시 device 인자 없이
            else:
                # CPU 사용 시 직접 로드
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    **model_kwargs
                )
                self.model = self.model.to(self.device)
                pipeline_device = -1  # CPU는 -1
            
            # Pipeline 생성
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True
            }
            
            # device_map 사용 시 device 인자 제거
            if pipeline_device is not None:
                pipeline_kwargs["device"] = pipeline_device
            
            self.pipeline = pipeline("text-generation", **pipeline_kwargs)
            
            print("✅ 모델 로드 완료!")
            
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            print("\n대안:")
            print("1. 더 작은 모델 사용: 'microsoft/DialoGPT-small'")
            print("2. 메모리 부족 시: load_in_8bit=True 옵션 사용")
            print("3. CPU 사용 시: 더 작은 모델 권장")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """프롬프트를 받아 LLM으로 생성"""
        if self.provider in ["llama", "local"]:
            try:
                # Llama 모델용 프롬프트 포맷
                if "Llama-2" in self.model_name or "llama-2" in self.model_name.lower():
                    formatted_prompt = f"[INST] {prompt} [/INST]"
                else:
                    formatted_prompt = prompt
                
                # Pipeline 사용
                if self.pipeline:
                    result = self.pipeline(
                        formatted_prompt,
                        max_new_tokens=min(max_tokens, 200),
                        num_return_sequences=1,
                        truncation=True
                    )
                    generated_text = result[0]["generated_text"]
                    # 프롬프트 부분 제거
                    if formatted_prompt in generated_text:
                        response = generated_text.split(formatted_prompt)[-1].strip()
                    else:
                        response = generated_text.strip()
                    return response
                else:
                    # 직접 토크나이징
                    inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
                    if self.device == "cuda":
                        inputs = inputs.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=min(max_tokens, 200),
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # 프롬프트 부분 제거
                    if formatted_prompt in response:
                        response = response.split(formatted_prompt)[-1].strip()
                    return response
                    
            except Exception as e:
                return f"[LLM 에러: {str(e)}]"
        
        elif self.provider == "openai":
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name if self.model_name else "gpt-3.5-turbo",
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
        
        if self.llm is None:
            # LLM 없으면 키워드 매칭으로 폴백
            return self._keyword_search(query)
        
        # LLM으로 검색 쿼리 개선
        improved_query = self.llm.generate(
            f"다음 질문의 핵심 키워드를 추출하세요: {query}\n키워드:",
            max_tokens=50
        )
        
        # 의미적으로 유사한 질문 찾기
        results = []
        query_lower = query.lower()
        
        for qa_pair in self.qa_pairs:
            question = qa_pair.get("question", "").lower()
            
            # LLM으로 관련성 판단
            relevance_prompt = f"""
다음 두 질문이 관련이 있는지 판단하세요.

질문1: {query}
질문2: {qa_pair.get('question', '')}

관련성이 있으면 'yes', 없으면 'no'만 답변하세요:"""
            
            relevance = self.llm.generate(relevance_prompt, max_tokens=10).lower()
            
            if 'yes' in relevance:
                results.append({
                    "source": "hotpotqa",
                    "data": qa_pair,
                    "relevance_score": 1.0
                })
        
        return results[:5]
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (폴백)"""
        results = []
        query_lower = query.lower()
        
        for qa_pair in self.qa_pairs:
            question = qa_pair.get("question", "").lower()
            answer = qa_pair.get("answer", "").lower()
            
            query_words = set(query_lower.split())
            question_words = set(question.split())
            
            if query_words & question_words:
                results.append({
                    "source": "hotpotqa",
                    "data": qa_pair
                })
        
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


def test_with_llm_example(model_name: Optional[str] = None, use_8bit: bool = False):
    """Llama LLM을 사용한 테스트 예시"""
    print("="*80)
    print("Llama를 사용한 HotpotQA 테스트")
    print("="*80)
    
    # 기본 모델명 (더 작은 모델부터 시도)
    default_models = [
        "microsoft/DialoGPT-medium",  # 작고 빠름 (약 350M)
        "meta-llama/Llama-2-7b-chat-hf",  # Llama 2 (약 7B)
        "meta-llama/Llama-2-13b-chat-hf",  # Llama 2 (약 13B)
    ]
    
    model_to_use = model_name or default_models[0]  # 기본값: 작은 모델
    
    # LLM 초기화
    try:
        print(f"\n모델 로드 시도: {model_to_use}")
        print("더 큰 모델을 사용하려면 --model 옵션으로 변경하세요.")
        
        llm = LLMInterface(
            provider="llama",
            model_name=model_to_use,
            device="auto",
            load_in_8bit=use_8bit
        )
        print("✅ Llama 모델 초기화 완료!")
    except Exception as e:
        print(f"\n❌ 모델 로드 실패: {str(e)}")
        print("\n해결 방법:")
        print("1. 더 작은 모델 시도:")
        print("   python hotpot_qa_with_llm.py --model microsoft/DialoGPT-small")
        print("2. 8-bit 양자화 사용 (GPU 메모리 절약):")
        print("   python hotpot_qa_with_llm.py --use-8bit")
        print("3. LLM 없이 테스트:")
        print("   python hotpot_qa_integration.py")
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HotpotQA 테스트 (Llama 사용)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="사용할 모델명 (기본: microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="8-bit 양자화 사용 (GPU 메모리 절약)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Llama 모델을 사용하는 HotpotQA 테스트")
    print("="*80)
    print("\n필요한 라이브러리:")
    print("  pip install transformers torch accelerate")
    print("\n권장 모델 (작은 것부터):")
    print("  1. microsoft/DialoGPT-medium (약 350M) - 빠름, 기본값")
    print("  2. meta-llama/Llama-2-7b-chat-hf (7B) - 더 정확, GPU 필요")
    print("  3. meta-llama/Llama-2-13b-chat-hf (13B) - 매우 정확, 강력한 GPU 필요")
    print("="*80)
    
    test_with_llm_example(model_name=args.model, use_8bit=args.use_8bit)

