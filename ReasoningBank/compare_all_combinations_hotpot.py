"""
HotpotQA + OpenAI를 사용하여 ReAct, Reflexion, ReasoningBank의 모든 조합 비교
"""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from datetime import datetime

from react_reflexion_reasoningbank import (
    ReActReflexionReasoningBank,
    ReasoningBank,
    Reflexion,
    ReActAgent
)
from hotpot_qa_with_llm import LLMInterface, HotpotQAReasoningBankWithLLM, ReActWithLLM


class AllCombinationsComparison:
    """모든 조합 비교 테스트"""
    
    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        self.llm = None
        self.hotpot_bank = None
        self.miniworld_bank = None
        self.reflexion = None
        self.results = {}
        
    def setup(self) -> bool:
        """초기 설정"""
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            return False
        
        # LLM 초기화
        try:
            self.llm = LLMInterface(
                provider="openai",
                model_name="gpt-3.5-turbo",
                api_key=api_key
            )
            print("✅ OpenAI API 초기화 완료")
        except Exception as e:
            print(f"❌ LLM 초기화 실패: {str(e)}")
            return False
        
        # HotpotQA 데이터셋 로드
        self.hotpot_bank = HotpotQAReasoningBankWithLLM(llm=self.llm)
        if not self.hotpot_bank.load_dataset(
            split="validation",
            subset="distractor",
            num_samples=self.num_samples
        ):
            return False
        
        # Miniworld 데이터셋 (전체 시스템용)
        base_path = os.path.dirname(os.path.abspath(__file__))
        bio_path = os.path.join(base_path, "miniworld_bio.csv")
        corpus_path = os.path.join(base_path, "miniworld_corpus.csv")
        world_path = os.path.join(base_path, "miniworld.csv")
        self.miniworld_bank = ReasoningBank(bio_path, corpus_path, world_path)
        
        # Reflexion 초기화
        self.reflexion = Reflexion()
        
        return True
    
    def test_react_only(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct만 사용 (HotpotQA)"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # ReAct 에이전트 (LLM 사용, ReasoningBank 없이)
        agent = ReActWithLLM(self.llm, self.hotpot_bank)
        
        steps = []
        
        # Step 1: Reasoning만 (검색 없이)
        step1 = agent.step(1, 
            f"질문 '{question}'에 답하기 위해 추론해야 합니다.",
            "reason",
            {"question": question, "context": ""}
        )
        steps.append(step1)
        
        # Step 2: 답변 생성
        step2 = agent.step(2,
            "추론을 바탕으로 답변을 생성합니다.",
            "answer",
            {"question": question, "context": step1['observation']}
        )
        steps.append(step2)
        
        final_answer = step2['observation']
        success = self._check_answer_correct(final_answer, correct_answer)
        
        return {
            "success": success,
            "steps_count": len(steps),
            "answer": final_answer,
            "correct_answer": correct_answer,
            "steps": [f"Step {s['step_num']}: {s['action']}" for s in steps],
            "search_count": 0,
            "reflection_count": 0
        }
    
    def test_reflexion_only(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Reflexion만 사용"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # 과거 메모리 추가
        self.reflexion.add_memory(1, "multi-hop 질문", True, None, "여러 정보를 종합해야 함")
        
        memories = self.reflexion.get_relevant_memories(question, top_k=3)
        
        # LLM으로 메모리 기반 답변 생성
        if memories:
            context = "\n".join([m.lesson_learned for m in memories])
            answer = self.llm.generate_answer(question, f"과거 경험: {context}")
        else:
            answer = "관련 과거 경험이 없습니다."
        
        success = self._check_answer_correct(answer, correct_answer)
        
        return {
            "success": success,
            "steps_count": 1,
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": ["1. Reflexion 메모리 검색"],
            "search_count": 0,
            "reflection_count": len(memories)
        }
    
    def test_reasoningbank_only(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReasoningBank만 사용"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # HotpotQA 검색
        keywords = question.split()[:2]
        query = " ".join(keywords)
        results = self.hotpot_bank.search_with_llm(query)
        
        if results:
            # 검색 결과를 LLM으로 분석
            context = "\n".join([
                f"Q: {r['data'].get('question', '')}\nA: {r['data'].get('answer', '')}"
                for r in results[:3]
            ])
            answer = self.llm.generate_answer(question, context)
        else:
            answer = "검색 결과를 찾을 수 없습니다."
        
        success = self._check_answer_correct(answer, correct_answer)
        
        return {
            "success": success,
            "steps_count": 1,
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": ["1. ReasoningBank 검색"],
            "search_count": len(results),
            "reflection_count": 0
        }
    
    def test_react_reflexion(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct + Reflexion"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # Reflexion 메모리
        self.reflexion.add_memory(1, "multi-hop 질문", True, None, "단계별 추론 필요")
        memories = self.reflexion.get_relevant_memories(question, top_k=2)
        
        agent = ReActWithLLM(self.llm, self.hotpot_bank)
        steps = []
        
        # Step 1: Reflexion 기반 Reasoning
        reflection_context = "\n".join([m.lesson_learned for m in memories]) if memories else ""
        step1 = agent.step(1,
            "과거 경험을 바탕으로 추론합니다.",
            "reason",
            {"question": question, "context": reflection_context}
        )
        steps.append(step1)
        
        # Step 2: 답변 생성
        step2 = agent.step(2,
            "추론 결과를 바탕으로 답변을 생성합니다.",
            "answer",
            {"question": question, "context": step1['observation']}
        )
        steps.append(step2)
        
        final_answer = step2['observation']
        success = self._check_answer_correct(final_answer, correct_answer)
        
        return {
            "success": success,
            "steps_count": len(steps),
            "answer": final_answer,
            "correct_answer": correct_answer,
            "steps": [f"Step {s['step_num']}: {s['action']}" for s in steps],
            "search_count": 0,
            "reflection_count": len(memories)
        }
    
    def test_react_reasoningbank(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct + ReasoningBank"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        agent = ReActWithLLM(self.llm, self.hotpot_bank)
        steps = []
        
        # Step 1: 검색
        keywords = question.split()[:2]
        query = " ".join(keywords)
        step1 = agent.step(1,
            "질문에 답하기 위해 정보를 검색합니다.",
            "search",
            {"query": query}
        )
        steps.append(step1)
        
        # Step 2: 추론
        step2 = agent.step(2,
            "검색 결과를 바탕으로 추론합니다.",
            "reason",
            {"question": question, "context": step1['observation']}
        )
        steps.append(step2)
        
        # Step 3: 답변 생성
        step3 = agent.step(3,
            "최종 답변을 생성합니다.",
            "answer",
            {"question": question, "context": step1['observation'] + "\n" + step2['observation']}
        )
        steps.append(step3)
        
        final_answer = step3['observation']
        success = self._check_answer_correct(final_answer, correct_answer)
        
        return {
            "success": success,
            "steps_count": len(steps),
            "answer": final_answer,
            "correct_answer": correct_answer,
            "steps": [f"Step {s['step_num']}: {s['action']}" for s in steps],
            "search_count": 1,
            "reflection_count": 0
        }
    
    def test_reflexion_reasoningbank(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Reflexion + ReasoningBank"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # Reflexion 메모리
        self.reflexion.add_memory(1, "multi-hop 질문", True, None, "검색 후 종합 분석 필요")
        memories = self.reflexion.get_relevant_memories(question, top_k=2)
        
        # ReasoningBank 검색
        keywords = question.split()[:2]
        query = " ".join(keywords)
        search_results = self.hotpot_bank.search_with_llm(query)
        
        # Reflexion 가이드 + 검색 결과를 LLM으로 분석
        reflection_context = "\n".join([m.lesson_learned for m in memories]) if memories else ""
        search_context = "\n".join([
            f"Q: {r['data'].get('question', '')}\nA: {r['data'].get('answer', '')}"
            for r in search_results[:3]
        ])
        
        combined_context = f"과거 경험: {reflection_context}\n\n검색 결과:\n{search_context}"
        answer = self.llm.generate_answer(question, combined_context)
        
        success = self._check_answer_correct(answer, correct_answer)
        
        return {
            "success": success,
            "steps_count": 2,
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": ["1. Reflexion 메모리 검색", "2. ReasoningBank 검색 및 분석"],
            "search_count": len(search_results),
            "reflection_count": len(memories)
        }
    
    def test_full_system(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct + Reflexion + ReasoningBank (전체 시스템)"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # Reflexion 메모리 추가
        self.reflexion.add_memory(1, "multi-hop 질문", True, None, "검색 후 추론하여 답변 생성 필요")
        memories = self.reflexion.get_relevant_memories(question, top_k=2)
        
        # ReAct 에이전트 (HotpotQA ReasoningBank 사용)
        agent = ReActWithLLM(self.llm, self.hotpot_bank)
        steps = []
        
        try:
            # Step 1: ReasoningBank 검색
            keywords = question.split()[:3]  # 더 많은 키워드 사용
            query = " ".join(keywords)
            step1 = agent.step(1,
                f"질문 '{question}'에 답하기 위해 HotpotQA에서 관련 정보를 검색합니다.",
                "search",
                {"query": query}
            )
            steps.append(step1)
            
            # Step 2: Reflexion 메모리 확인 및 추론
            reflection_context = "\n".join([m.lesson_learned for m in memories]) if memories else ""
            step2 = agent.step(2,
                "검색 결과와 과거 경험을 바탕으로 추론합니다.",
                "reason",
                {"question": question, "context": step1['observation'] + "\n과거 경험: " + reflection_context}
            )
            steps.append(step2)
            
            # Step 3: 최종 답변 생성
            combined_context = step1['observation'] + "\n" + step2['observation']
            step3 = agent.step(3,
                "추론 결과를 바탕으로 최종 답변을 생성합니다.",
                "answer",
                {"question": question, "context": combined_context}
            )
            steps.append(step3)
            
            final_answer = step3['observation']
            success = self._check_answer_correct(final_answer, correct_answer)
            
            return {
                "success": success,
                "steps_count": len(steps),
                "answer": final_answer[:200] if len(final_answer) > 200 else final_answer,
                "correct_answer": correct_answer,
                "steps": [f"Step {s['step_num']}: {s['action']}" for s in steps],
                "search_count": 1,
                "reflection_count": len(memories)
            }
        except Exception as e:
            return {
                "success": False,
                "steps_count": len(steps),
                "answer": f"에러: {str(e)}",
                "correct_answer": correct_answer,
                "steps": [f"Step {s['step_num']}: {s['action']}" for s in steps] if steps else [],
                "search_count": sum(1 for s in steps if s['action'] == 'search'),
                "reflection_count": len(memories)
            }
    
    def _check_answer_correct(self, generated: str, correct: str) -> bool:
        """답변 정확도 체크"""
        generated_lower = generated.lower()
        correct_lower = correct.lower()
        
        # "답변:" 프리픽스 제거
        if "답변:" in generated_lower:
            generated_lower = generated_lower.split("답변:")[-1].strip()
        
        # 직접 매칭
        if correct_lower in generated_lower or generated_lower in correct_lower:
            return True
        
        # yes/no 체크
        if correct_lower in ["yes", "no"]:
            if (correct_lower in generated_lower or
                ("네" in generated_lower and correct_lower == "yes") or
                ("같은" in generated_lower and correct_lower == "yes") or
                ("아니" in generated_lower and correct_lower == "no")):
                return True
        
        return False
    
    def run_comparison(self):
        """모든 조합 비교 실행"""
        if not self.setup():
            return None
        
        print("\n" + "="*80)
        print("모든 조합 비교 테스트 시작")
        print("="*80)
        
        combinations = {
            "ReAct만": self.test_react_only,
            "Reflexion만": self.test_reflexion_only,
            "ReasoningBank만": self.test_reasoningbank_only,
            "ReAct + Reflexion": self.test_react_reflexion,
            "ReAct + ReasoningBank": self.test_react_reasoningbank,
            "Reflexion + ReasoningBank": self.test_reflexion_reasoningbank,
            "ReAct + Reflexion + ReasoningBank": self.test_full_system
        }
        
        qa_pairs = self.hotpot_bank.get_qa_pairs()
        all_results = {}
        
        for combo_name, test_func in combinations.items():
            print(f"\n{'='*80}")
            print(f"테스트: {combo_name}")
            print(f"{'='*80}")
            
            combo_results = []
            for idx, qa_pair in enumerate(qa_pairs[:self.num_samples], 1):
                question = qa_pair["question"]
                print(f"\n[{idx}/{len(qa_pairs[:self.num_samples])}] {question[:60]}...")
                
                try:
                    result = test_func(qa_pair)
                    result["question"] = question
                    combo_results.append(result)
                    status = "✅" if result['success'] else "❌"
                    print(f"  {status} 성공: {result['success']}, 스텝: {result['steps_count']}")
                except Exception as e:
                    print(f"  ❌ 에러: {str(e)}")
                    combo_results.append({
                        "question": question,
                        "success": False,
                        "steps_count": 0,
                        "answer": f"에러: {str(e)}",
                        "correct_answer": qa_pair.get("answer", ""),
                        "steps": [],
                        "search_count": 0,
                        "reflection_count": 0
                    })
            
            all_results[combo_name] = combo_results
        
        self.results = all_results
        return all_results
    
    def generate_table(self, output_file: str = "hotpot_all_combinations_table.md"):
        """비교 표 생성"""
        if not self.results:
            print("결과가 없습니다.")
            return
        
        markdown = "# HotpotQA + OpenAI를 사용한 모든 조합 비교\n\n"
        markdown += f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 종합 비교표
        markdown += "## 종합 비교표\n\n"
        markdown += "| 조합 | 평균 성공률 | 평균 스텝 수 | 평균 검색 횟수 | 평균 Reflection 횟수 |\n"
        markdown += "|------|------------|------------|--------------|-------------------|\n"
        
        for combo_name, results in self.results.items():
            if results:
                success_count = sum(1 for r in results if r.get('success', False))
                avg_success = success_count / len(results) * 100
                avg_steps = sum(r['steps_count'] for r in results) / len(results)
                avg_search = sum(r.get('search_count', 0) for r in results) / len(results)
                avg_reflection = sum(r.get('reflection_count', 0) for r in results) / len(results)
                
                markdown += f"| {combo_name} | {avg_success:.1f}% | {avg_steps:.1f} | {avg_search:.1f} | {avg_reflection:.1f} |\n"
        
        # 질문별 상세 비교
        markdown += "\n## 질문별 상세 비교\n\n"
        
        if self.results:
            sample_results = list(self.results.values())[0]
            
            for idx, result in enumerate(sample_results, 1):
                markdown += f"### 질문 {idx}: {result.get('question', '')[:100]}...\n\n"
                markdown += "| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |\n"
                markdown += "|------|------|---------|------|------------|----------|\n"
                
                for combo_name, results in self.results.items():
                    if idx <= len(results):
                        r = results[idx - 1]
                        success_mark = "✅" if r.get('success', False) else "❌"
                        answer_summary = (r.get('answer', '')[:50] + "...") if len(r.get('answer', '')) > 50 else r.get('answer', '')
                        markdown += f"| {combo_name} | {success_mark} | {r['steps_count']} | {r.get('search_count', 0)} | {r.get('reflection_count', 0)} | {answer_summary} |\n"
                
                markdown += "\n"
        
        # JSON 저장
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Markdown 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"\n비교 표가 생성되었습니다:")
        print(f"  - {output_file}")
        print(f"  - {json_file}")
        
        return markdown


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모든 조합 비교 테스트")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="테스트할 질문 개수 (기본: 5)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("HotpotQA + OpenAI를 사용한 모든 조합 비교 테스트")
    print("="*80)
    
    comparison = AllCombinationsComparison(num_samples=args.num_samples)
    results = comparison.run_comparison()
    
    if results:
        comparison.generate_table()
        
        # 요약 출력
        print("\n" + "="*80)
        print("종합 결과 요약")
        print("="*80)
        
        for combo_name, combo_results in results.items():
            if combo_results:
                success_count = sum(1 for r in combo_results if r.get('success', False))
                avg_steps = sum(r['steps_count'] for r in combo_results) / len(combo_results)
                print(f"\n{combo_name}:")
                print(f"  성공률: {success_count}/{len(combo_results)} ({success_count/len(combo_results)*100:.1f}%)")
                print(f"  평균 스텝: {avg_steps:.1f}")
    else:
        print("테스트 실행 실패")


if __name__ == "__main__":
    main()

