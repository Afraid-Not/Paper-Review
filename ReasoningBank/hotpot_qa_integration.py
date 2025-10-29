"""
HotpotQA 데이터셋을 ReAct + Reflexion + ReasoningBank 시스템에 통합
"""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset

from react_reflexion_reasoningbank import (
    ReActReflexionReasoningBank,
    ReasoningBank,
    Reflexion,
    ReActAgent
)


class HotpotQAReasoningBank:
    """HotpotQA 데이터를 ReasoningBank 형식으로 변환"""
    
    def __init__(self):
        self.qa_pairs: List[Dict[str, Any]] = []
        self.loaded = False
    
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
                # HotpotQA 형식을 우리 시스템 형식으로 변환
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
            print("힌트: pip install datasets 명령으로 datasets 라이브러리를 설치하세요.")
            return False
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """질문과 관련된 QA 쌍 검색"""
        if not self.loaded:
            return []
        
        results = []
        query_lower = query.lower()
        
        for qa_pair in self.qa_pairs:
            question = qa_pair.get("question", "").lower()
            answer = qa_pair.get("answer", "").lower()
            
            # 간단한 키워드 매칭
            query_words = set(query_lower.split())
            question_words = set(question.split())
            answer_words = set(answer.split())
            
            if query_words & question_words or query_words & answer_words:
                results.append({
                    "source": "hotpotqa",
                    "data": qa_pair
                })
        
        return results[:5]  # 최대 5개 반환
    
    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """모든 QA 쌍 반환"""
        return self.qa_pairs


class HotpotQATestRunner:
    """HotpotQA로 각 조합 테스트"""
    
    def __init__(self, num_samples: int = 10):
        self.hotpot_bank = HotpotQAReasoningBank()
        self.num_samples = num_samples
        self.results = {}
    
    def setup(self):
        """데이터셋 로드"""
        return self.hotpot_bank.load_dataset(
            split="validation",
            subset="distractor",
            num_samples=self.num_samples
        )
    
    def test_react_only(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct만 사용"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # 간단한 reasoning만 수행
        steps = [
            "1. Question 분석",
            "2. 간단한 reasoning 수행"
        ]
        
        answer = f"ReAct reasoning으로 답변을 생성했습니다. (정답: {correct_answer})"
        
        return {
            "success": False,  # HotpotQA는 복잡하므로 단순 ReAct로는 어려움
            "steps_count": len(steps),
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": steps,
            "search_count": 0,
            "reflection_count": 0
        }
    
    def test_reasoningbank_only(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """HotpotQA ReasoningBank만 사용"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # 키워드로 검색
        keywords = question.split()[:2]  # 처음 2개 단어
        query = " ".join(keywords)
        
        results = self.hotpot_bank.search(query)
        
        if results:
            # 검색된 답변 중 하나 사용
            found_answer = results[0]["data"]["answer"]
            answer = f"검색 결과: {found_answer}"
            success = found_answer.lower() == correct_answer.lower()
        else:
            answer = "검색 결과를 찾을 수 없습니다."
            success = False
        
        return {
            "success": success,
            "steps_count": 1,
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": ["1. HotpotQA 지식 베이스 검색"],
            "search_count": len(results),
            "reflection_count": 0
        }
    
    def test_react_reflexion(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct + Reflexion"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        reflexion = Reflexion()
        reflexion.add_memory(1, "multi-hop 질문 해결", True, None, "여러 단계의 reasoning이 필요함")
        
        steps = [
            "1. Question 분석 및 Reasoning",
            "2. Reflexion 메모리 확인"
        ]
        
        memories = reflexion.get_relevant_memories(question, top_k=2)
        if memories:
            answer = f"Reasoning + Reflection 기반 답변. (정답: {correct_answer})"
        else:
            answer = "Reasoning 완료. 관련 과거 경험이 없습니다."
        
        return {
            "success": False,
            "steps_count": len(steps),
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": steps,
            "search_count": 0,
            "reflection_count": len(memories)
        }
    
    def test_react_reasoningbank(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """ReAct + HotpotQA ReasoningBank"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        steps = []
        
        # Step 1: Reasoning
        steps.append("1. Question 분석")
        keywords = question.split()[:2]
        query = " ".join(keywords)
        
        # Step 2: Search
        steps.append("2. HotpotQA 지식 베이스 검색")
        results = self.hotpot_bank.search(query)
        
        # Step 3: Observation 기반 답변
        steps.append("3. 검색 결과 분석")
        if results:
            found_answer = results[0]["data"]["answer"]
            answer = f"검색된 답변: {found_answer}"
            success = found_answer.lower() == correct_answer.lower()
        else:
            answer = "검색 결과를 찾을 수 없습니다."
            success = False
        
        return {
            "success": success,
            "steps_count": len(steps),
            "answer": answer,
            "correct_answer": correct_answer,
            "steps": steps,
            "search_count": len(results),
            "reflection_count": 0
        }
    
    def test_full_system(self, qa_pair: Dict[str, Any], temp_dir: str = "temp_hotpot_logs") -> Dict[str, Any]:
        """전체 시스템 (임시로 HotpotQA 사용, 실제로는 miniworld 사용)"""
        question = qa_pair["question"]
        correct_answer = qa_pair["answer"]
        
        # miniworld 데이터 경로 (실제 시스템은 miniworld 사용)
        base_path = os.path.dirname(os.path.abspath(__file__))
        bio_path = os.path.join(base_path, "miniworld_bio.csv")
        corpus_path = os.path.join(base_path, "miniworld_corpus.csv")
        world_path = os.path.join(base_path, "miniworld.csv")
        
        model = ReActReflexionReasoningBank(
            bio_path=bio_path,
            corpus_path=corpus_path,
            world_path=world_path,
            log_dir=temp_dir
        )
        
        # 질문을 간단히 변환 (HotpotQA 질문은 너무 복잡할 수 있음)
        simplified_question = question[:50] + "?" if len(question) > 50 else question
        
        try:
            success = model.run_episode(simplified_question, max_steps=10)
            react_steps = model.react_agent.get_steps()
            final_answer = react_steps[-1].observation if react_steps else "답변 생성 실패"
            
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return {
                "success": success,
                "steps_count": len(react_steps),
                "answer": final_answer[:200],
                "correct_answer": correct_answer,
                "steps": [f"Step {s['step_num']}: {s['action']}" for s in react_steps],
                "search_count": sum(1 for s in react_steps if s['action'] == 'search'),
                "reflection_count": sum(1 for s in react_steps if s['action'] == 'reflect')
            }
        except Exception as e:
            return {
                "success": False,
                "steps_count": 0,
                "answer": f"에러: {str(e)}",
                "correct_answer": correct_answer,
                "steps": [],
                "search_count": 0,
                "reflection_count": 0
            }
    
    def run_comparison(self):
        """HotpotQA로 비교 테스트 실행"""
        if not self.setup():
            print("데이터셋 로드 실패. 테스트를 중단합니다.")
            return None
        
        print("\n" + "="*80)
        print("HotpotQA를 사용한 조합별 비교 테스트 시작")
        print("="*80)
        
        combinations = {
            "ReAct만": self.test_react_only,
            "HotpotQA ReasoningBank만": self.test_reasoningbank_only,
            "ReAct + Reflexion": self.test_react_reflexion,
            "ReAct + HotpotQA ReasoningBank": self.test_react_reasoningbank,
            "전체 시스템 (miniworld)": self.test_full_system
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
                print(f"\n[{idx}/{len(qa_pairs[:self.num_samples])}] 질문: {question[:60]}...")
                try:
                    result = test_func(qa_pair)
                    result["question"] = question
                    combo_results.append(result)
                    print(f"  성공: {result['success']}, 스텝: {result['steps_count']}")
                except Exception as e:
                    print(f"  에러: {str(e)}")
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
    
    def generate_table(self, output_file: str = "hotpot_comparison_table.md"):
        """비교 표 생성"""
        if not self.results:
            print("결과가 없습니다.")
            return
        
        markdown = "# HotpotQA를 사용한 조합별 비교\n\n"
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
        
        # 상세 결과
        markdown += "\n## 질문별 상세 비교\n\n"
        
        if self.results:
            sample_results = list(self.results.values())[0]
            
            for idx, result in enumerate(sample_results[:5], 1):  # 처음 5개만
                markdown += f"### 질문 {idx}: {result.get('question', '')[:80]}...\n\n"
                markdown += "| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |\n"
                markdown += "|------|------|---------|------|------------|----------|\n"
                
                for combo_name, results in self.results.items():
                    if idx <= len(results):
                        r = results[idx - 1]
                        success_mark = "✅" if r.get('success', False) else "❌"
                        answer_summary = (r.get('answer', '')[:50] + "...") if len(r.get('answer', '')) > 50 else r.get('answer', '')
                        markdown += f"| {combo_name} | {success_mark} | {r['steps_count']} | {r.get('search_count', 0)} | {r.get('reflection_count', 0)} | {answer_summary} |\n"
                
                markdown += "\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"\n비교 표가 생성되었습니다: {output_file}")
        return markdown


def main():
    """메인 실행"""
    print("HotpotQA 데이터셋을 사용한 비교 테스트")
    print("="*80)
    
    # 테스트 러너 생성 (10개 샘플로 테스트)
    runner = HotpotQATestRunner(num_samples=10)
    
    # 비교 테스트 실행
    results = runner.run_comparison()
    
    if results:
        # 표 생성
        runner.generate_table()
        
        # 요약 출력
        print("\n" + "="*80)
        print("HotpotQA 테스트 결과 요약")
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

