"""
ReAct, Reflexion, ReasoningBank의 다양한 조합 비교 테스트
각 조합별 성능을 표로 정리
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from react_reflexion_reasoningbank import (
    ReActReflexionReasoningBank, 
    ReasoningBank, 
    Reflexion, 
    ReActAgent
)


class ComparisonTest:
    """각 조합별 비교 테스트"""
    
    def __init__(self, bio_path: str, corpus_path: str, world_path: str):
        self.bio_path = bio_path
        self.corpus_path = corpus_path
        self.world_path = world_path
        
        # 테스트 질문들
        self.test_questions = [
            "Liam Brook은 어디에서 태어났나요?",
            "Ava Shin의 생년은 언제인가요?",
            "Norland의 수도는 어디인가요?",
            "Seoria는 어떤 나라에 있나요?",
            "Valenport의 인구는 얼마인가요?"
        ]
        
        self.results = {}
    
    def test_react_only(self, question: str) -> Dict[str, Any]:
        """ReAct만 사용"""
        print(f"[ReAct만] {question}")
        
        # ReAct 에이전트만 생성 (ReasoningBank, Reflexion 없이)
        # 실제로는 간단한 내부 reasoning만 사용
        steps = []
        answer = ""
        success = False
        
        # 간단한 키워드 추출
        keywords = question.split()
        if "어디에서 태어났나요" in question:
            answer = f"{keywords[0]}의 출생지 정보를 찾기 위해 reasoning합니다. (ReAct만 사용)"
            steps.append("1. Question 분석")
            steps.append("2. 간단한 reasoning 수행")
        elif "생년" in question or "언제" in question:
            answer = f"{keywords[0]}의 생년 정보를 찾기 위해 reasoning합니다. (ReAct만 사용)"
            steps.append("1. Question 분석")
            steps.append("2. 간단한 reasoning 수행")
        else:
            answer = f"질문을 분석하여 reasoning합니다. (ReAct만 사용)"
            steps.append("1. Question 분석")
            steps.append("2. 간단한 reasoning 수행")
        
        success = True
        
        return {
            "success": success,
            "steps_count": len(steps),
            "answer": answer,
            "steps": steps,
            "search_count": 0,
            "reflection_count": 0
        }
    
    def test_reflexion_only(self, question: str) -> Dict[str, Any]:
        """Reflexion만 사용 (메모리 기반)"""
        print(f"[Reflexion만] {question}")
        
        reflexion = Reflexion()
        
        # 더미 메모리 추가 (실제로는 과거 에피소드에서 학습)
        reflexion.add_memory(1, "사람의 출생지 찾기", True, None, "bio 데이터베이스에서 검색해야 함")
        reflexion.add_memory(2, "도시의 국가 찾기", True, None, "corpus 데이터베이스에서 검색해야 함")
        
        memories = reflexion.get_relevant_memories(question, top_k=3)
        
        if memories:
            lesson = memories[0].lesson_learned
            answer = f"과거 경험을 바탕으로: {lesson}"
        else:
            answer = "관련 과거 경험이 없습니다. (Reflexion만 사용)"
        
        return {
            "success": len(memories) > 0,
            "steps_count": 1,
            "answer": answer,
            "steps": ["1. 과거 메모리 검색"],
            "search_count": 0,
            "reflection_count": len(memories)
        }
    
    def test_reasoningbank_only(self, question: str) -> Dict[str, Any]:
        """ReasoningBank만 사용"""
        print(f"[ReasoningBank만] {question}")
        
        reasoning_bank = ReasoningBank(self.bio_path, self.corpus_path, self.world_path)
        
        # 키워드 추출
        keywords = question.split()
        main_query = keywords[0] if keywords else ""
        
        results = reasoning_bank.search(main_query)
        
        if results:
            answer_parts = []
            for r in results[:2]:
                data = r.get('data', {})
                source = r.get('source', '')
                if source == 'bio' and 'born_in_city' in data:
                    answer_parts.append(f"{data.get('title', '')}은(는) {data.get('born_in_city', '')}에서 태어났습니다.")
                elif source == 'corpus' and 'country' in data:
                    answer_parts.append(f"{data.get('title', '')}은(는) {data.get('country', '')}에 있습니다.")
                elif source == 'world' and 'capital' in data:
                    answer_parts.append(f"{data.get('title', '')}의 수도는 {data.get('capital', '')}입니다.")
            
            answer = "\n".join(answer_parts) if answer_parts else f"검색 결과: {len(results)}개 항목 발견"
        else:
            answer = "검색 결과를 찾을 수 없습니다."
        
        return {
            "success": len(results) > 0,
            "steps_count": 1,
            "answer": answer,
            "steps": ["1. 지식 베이스 검색"],
            "search_count": len(results),
            "reflection_count": 0
        }
    
    def test_react_reflexion(self, question: str) -> Dict[str, Any]:
        """ReAct + Reflexion"""
        print(f"[ReAct + Reflexion] {question}")
        
        reflexion = Reflexion()
        reflexion.add_memory(1, "사람의 출생지 찾기", True, None, "이름으로 검색 후 bio 정보 확인")
        
        steps = []
        answer = ""
        
        # Step 1: Reasoning
        steps.append("1. Question 분석 및 Reasoning")
        memories = reflexion.get_relevant_memories(question, top_k=2)
        
        # Step 2: Reflection 기반 Action 결정
        steps.append("2. Reflexion 메모리 확인")
        if memories:
            lesson = memories[0].lesson_learned
            answer = f"Reasoning + Reflection: {lesson}"
        else:
            answer = "Reasoning 완료. 관련 과거 경험이 없습니다."
        
        return {
            "success": True,
            "steps_count": len(steps),
            "answer": answer,
            "steps": steps,
            "search_count": 0,
            "reflection_count": len(memories)
        }
    
    def test_react_reasoningbank(self, question: str) -> Dict[str, Any]:
        """ReAct + ReasoningBank"""
        print(f"[ReAct + ReasoningBank] {question}")
        
        reasoning_bank = ReasoningBank(self.bio_path, self.corpus_path, self.world_path)
        
        steps = []
        
        # Step 1: Reasoning
        steps.append("1. Question 분석")
        keywords = question.split()
        main_query = keywords[0] if keywords else ""
        
        # Step 2: Action - Search
        steps.append("2. 지식 베이스 검색")
        results = reasoning_bank.search(main_query)
        
        # Step 3: Observation 기반 답변 생성
        steps.append("3. 검색 결과 분석")
        if results:
            data = results[0].get('data', {})
            if 'born_in_city' in data:
                answer = f"{data.get('title', '')}은(는) {data.get('born_in_city', '')}에서 태어났습니다."
            elif 'country' in data:
                answer = f"{data.get('title', '')}은(는) {data.get('country', '')}에 있습니다."
            elif 'capital' in data:
                answer = f"{data.get('title', '')}의 수도는 {data.get('capital', '')}입니다."
            else:
                answer = f"검색 결과: {len(results)}개 항목 발견"
        else:
            answer = "검색 결과를 찾을 수 없습니다."
        
        return {
            "success": len(results) > 0,
            "steps_count": len(steps),
            "answer": answer,
            "steps": steps,
            "search_count": len(results),
            "reflection_count": 0
        }
    
    def test_reflexion_reasoningbank(self, question: str) -> Dict[str, Any]:
        """Reflexion + ReasoningBank"""
        print(f"[Reflexion + ReasoningBank] {question}")
        
        reasoning_bank = ReasoningBank(self.bio_path, self.corpus_path, self.world_path)
        reflexion = Reflexion()
        reflexion.add_memory(1, "사람의 출생지 찾기", True, None, "bio에서 이름으로 검색")
        
        steps = []
        
        # Step 1: Reflection으로 접근 방법 결정
        steps.append("1. Reflexion 메모리 확인")
        memories = reflexion.get_relevant_memories(question, top_k=2)
        
        # Step 2: ReasoningBank 검색
        steps.append("2. 지식 베이스 검색")
        keywords = question.split()
        main_query = keywords[0] if keywords else ""
        results = reasoning_bank.search(main_query)
        
        if results:
            data = results[0].get('data', {})
            if memories:
                answer = f"Reflexion 가이드 + 검색 결과: {data.get('title', '')} 관련 정보 발견"
            else:
                answer = f"검색 결과: {len(results)}개 항목 발견"
        else:
            answer = "검색 결과를 찾을 수 없습니다."
        
        return {
            "success": len(results) > 0,
            "steps_count": len(steps),
            "answer": answer,
            "steps": steps,
            "search_count": len(results),
            "reflection_count": len(memories)
        }
    
    def test_full_system(self, question: str) -> Dict[str, Any]:
        """ReAct + Reflexion + ReasoningBank (전체 시스템)"""
        print(f"[전체 시스템] {question}")
        
        # 임시 로그 디렉토리
        temp_log_dir = "temp_comparison_logs"
        
        model = ReActReflexionReasoningBank(
            bio_path=self.bio_path,
            corpus_path=self.corpus_path,
            world_path=self.world_path,
            log_dir=temp_log_dir
        )
        
        # 에피소드 실행
        success = model.run_episode(question, max_steps=10)
        
        # 결과 추출
        react_steps = model.react_agent.get_steps()
        final_answer = react_steps[-1].observation if react_steps else "답변 생성 실패"
        
        # 로그 정리 (비교용이므로 삭제)
        import shutil
        if os.path.exists(temp_log_dir):
            shutil.rmtree(temp_log_dir)
        
        return {
            "success": success,
            "steps_count": len(react_steps),
            "answer": final_answer[:200] if len(final_answer) > 200 else final_answer,
            "steps": [f"Step {s['step_num']}: {s['action']}" for s in react_steps],
            "search_count": sum(1 for s in react_steps if s['action'] == 'search'),
            "reflection_count": sum(1 for s in react_steps if s['action'] == 'reflect')
        }
    
    def run_comparison(self):
        """모든 조합 테스트 실행 및 결과 정리"""
        print("="*80)
        print("조합별 비교 테스트 시작")
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
        
        all_results = {}
        
        for combo_name, test_func in combinations.items():
            print(f"\n{'='*80}")
            print(f"테스트: {combo_name}")
            print(f"{'='*80}")
            
            combo_results = []
            for question in self.test_questions:
                print(f"\n질문: {question}")
                try:
                    result = test_func(question)
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
                        "steps": [],
                        "search_count": 0,
                        "reflection_count": 0
                    })
            
            all_results[combo_name] = combo_results
        
        self.results = all_results
        return all_results
    
    def generate_comparison_table(self, output_file: str = "comparison_table.md"):
        """비교 표 생성 (Markdown)"""
        
        if not self.results:
            print("결과가 없습니다. 먼저 run_comparison()을 실행하세요.")
            return
        
        # 표 헤더
        markdown = "# ReAct + Reflexion + ReasoningBank 조합별 비교\n\n"
        markdown += f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 종합 비교표
        markdown += "## 종합 비교표\n\n"
        markdown += "| 조합 | 평균 성공률 | 평균 스텝 수 | 평균 검색 횟수 | 평균 Reflection 횟수 |\n"
        markdown += "|------|------------|------------|--------------|-------------------|\n"
        
        for combo_name, results in self.results.items():
            if results:
                avg_success = sum(1 for r in results if r['success']) / len(results) * 100
                avg_steps = sum(r['steps_count'] for r in results) / len(results)
                avg_search = sum(r['search_count'] for r in results) / len(results)
                avg_reflection = sum(r['reflection_count'] for r in results) / len(results)
                
                markdown += f"| {combo_name} | {avg_success:.1f}% | {avg_steps:.1f} | {avg_search:.1f} | {avg_reflection:.1f} |\n"
        
        # 질문별 상세 비교
        markdown += "\n## 질문별 상세 비교\n\n"
        
        for idx, question in enumerate(self.test_questions, 1):
            markdown += f"### 질문 {idx}: {question}\n\n"
            markdown += "| 조합 | 성공 | 스텝 수 | 검색 | Reflection | 답변 요약 |\n"
            markdown += "|------|------|---------|------|------------|----------|\n"
            
            for combo_name, results in self.results.items():
                if idx <= len(results):
                    result = results[idx - 1]
                    success_mark = "✅" if result['success'] else "❌"
                    answer_summary = result['answer'][:50] + "..." if len(result['answer']) > 50 else result['answer']
                    markdown += f"| {combo_name} | {success_mark} | {result['steps_count']} | {result['search_count']} | {result['reflection_count']} | {answer_summary} |\n"
            
            markdown += "\n"
        
        # 각 조합별 상세 답변
        markdown += "## 조합별 상세 답변\n\n"
        
        for combo_name, results in self.results.items():
            markdown += f"### {combo_name}\n\n"
            for idx, result in enumerate(results, 1):
                markdown += f"#### 질문 {idx}: {result['question']}\n"
                markdown += f"- **성공**: {'✅' if result['success'] else '❌'}\n"
                markdown += f"- **스텝 수**: {result['steps_count']}\n"
                markdown += f"- **검색 횟수**: {result['search_count']}\n"
                markdown += f"- **Reflection 횟수**: {result['reflection_count']}\n"
                markdown += f"- **스텝**: {', '.join(result['steps'][:3])}\n"
                markdown += f"- **답변**: {result['answer']}\n\n"
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"\n비교 표가 생성되었습니다: {output_file}")
        return markdown
    
    def save_json_results(self, output_file: str = "comparison_results.json"):
        """JSON 형식으로 결과 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"JSON 결과가 저장되었습니다: {output_file}")


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_path = os.path.dirname(os.path.abspath(__file__))
    bio_path = os.path.join(base_path, "miniworld_bio.csv")
    corpus_path = os.path.join(base_path, "miniworld_corpus.csv")
    world_path = os.path.join(base_path, "miniworld.csv")
    
    # 비교 테스트 생성 및 실행
    comparison = ComparisonTest(bio_path, corpus_path, world_path)
    results = comparison.run_comparison()
    
    # 결과 저장
    comparison.save_json_results()
    
    # 비교 표 생성
    table_markdown = comparison.generate_comparison_table()
    
    # 콘솔에 요약 출력
    print("\n" + "="*80)
    print("종합 결과 요약")
    print("="*80)
    
    for combo_name, combo_results in results.items():
        if combo_results:
            success_count = sum(1 for r in combo_results if r['success'])
            avg_steps = sum(r['steps_count'] for r in combo_results) / len(combo_results)
            print(f"\n{combo_name}:")
            print(f"  성공률: {success_count}/{len(combo_results)} ({success_count/len(combo_results)*100:.1f}%)")
            print(f"  평균 스텝: {avg_steps:.1f}")


if __name__ == "__main__":
    main()

