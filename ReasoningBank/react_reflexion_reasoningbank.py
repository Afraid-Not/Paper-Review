"""
ReAct + Reflexion + ReasoningBank 통합 강화학습 모델
- ReAct: Reasoning과 Acting 결합
- Reflexion: 이전 에피소드 메모리 저장
- ReasoningBank: 지식 베이스 저장 및 검색
"""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ReActStep:
    """ReAct의 한 단계를 나타내는 데이터 구조"""
    step_num: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: str


@dataclass
class ReflexionMemory:
    """Reflexion의 메모리 구조"""
    episode_num: int
    task_description: str
    success: bool
    failure_reason: Optional[str]
    lesson_learned: str
    timestamp: str


@dataclass
class BankMemory:
    """ReasoningBank의 메모리 구조"""
    query: str
    retrieved_knowledge: List[Dict[str, Any]]
    updated_knowledge: Optional[Dict[str, Any]]
    timestamp: str


class ReasoningBank:
    """지식 베이스를 저장하고 검색하는 시스템"""
    
    def __init__(self, bio_path: str, corpus_path: str, world_path: str):
        self.bio_path = bio_path
        self.corpus_path = corpus_path
        self.world_path = world_path
        
        # 데이터 로드
        self.bio_db = pd.read_csv(bio_path) if os.path.exists(bio_path) else pd.DataFrame()
        self.corpus_db = pd.read_csv(corpus_path) if os.path.exists(corpus_path) else pd.DataFrame()
        self.world_db = pd.read_csv(world_path) if os.path.exists(world_path) else pd.DataFrame()
        
        # 메모리 히스토리
        self.memory_history: List[BankMemory] = []
        
        print(f"[ReasoningBank] 로드된 데이터:")
        print(f"  - Bio: {len(self.bio_db)} 개")
        print(f"  - Corpus: {len(self.corpus_db)} 개")
        print(f"  - World: {len(self.world_db)} 개")
    
    def search(self, query: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """지식 베이스에서 검색"""
        results = []

        if not self.bio_db.empty:
            bio_matches = self.bio_db[
                self.bio_db['title'].str.contains(query, case=False, na=False) |
                self.bio_db['bio'].str.contains(query, case=False, na=False)
            ]
            for _, row in bio_matches.iterrows():
                results.append({
                    'source': 'bio',
                    'data': row.to_dict()
                })
        if not self.corpus_db.empty:
            corpus_matches = self.corpus_db[
                self.corpus_db['title'].str.contains(query, case=False, na=False) |
                self.corpus_db['summary'].str.contains(query, case=False, na=False)
            ]
            for _, row in corpus_matches.iterrows():
                results.append({
                    'source': 'corpus',
                    'data': row.to_dict()
                })
        if not self.world_db.empty:
            world_matches = self.world_db[
                self.world_db['title'].str.contains(query, case=False, na=False)
            ]
            for _, row in world_matches.iterrows():
                results.append({
                    'source': 'world',
                    'data': row.to_dict()
                })
        memory = BankMemory(
            query=query,
            retrieved_knowledge=results,
            updated_knowledge=None,
            timestamp=datetime.now().isoformat()
        )
        self.memory_history.append(memory)
        
        return results
    
    def update(self, key: str, field: str, value: Any, source: str = 'bio'):
        """지식 베이스 업데이트"""
        updated_knowledge = {
            'key': key,
            'field': field,
            'value': value,
            'source': source
        }
        
        # 실제 데이터프레임 업데이트
        if source == 'bio' and not self.bio_db.empty:
            mask = self.bio_db['title'] == key
            if mask.any():
                self.bio_db.loc[mask, field] = value
        
        elif source == 'corpus' and not self.corpus_db.empty:
            mask = self.corpus_db['title'] == key
            if mask.any():
                self.corpus_db.loc[mask, field] = value
        
        elif source == 'world' and not self.world_db.empty:
            mask = self.world_db['title'] == key
            if mask.any():
                self.world_db.loc[mask, field] = value
        
        # 메모리에 저장
        memory = BankMemory(
            query=f"UPDATE: {key}.{field}",
            retrieved_knowledge=[],
            updated_knowledge=updated_knowledge,
            timestamp=datetime.now().isoformat()
        )
        self.memory_history.append(memory)
    
    def get_memory_history(self) -> List[Dict[str, Any]]:
        """메모리 히스토리 반환"""
        return [asdict(memory) for memory in self.memory_history]


class Reflexion:
    """이전 에피소드에서 학습한 메모리를 저장하는 시스템"""
    
    def __init__(self):
        self.memories: List[ReflexionMemory] = []
    
    def add_memory(self, episode_num: int, task_description: str, 
                   success: bool, failure_reason: Optional[str] = None,
                   lesson_learned: str = ""):
        """새로운 메모리 추가"""
        memory = ReflexionMemory(
            episode_num=episode_num,
            task_description=task_description,
            success=success,
            failure_reason=failure_reason,
            lesson_learned=lesson_learned,
            timestamp=datetime.now().isoformat()
        )
        self.memories.append(memory)
    
    def get_relevant_memories(self, task_description: str, top_k: int = 3) -> List[ReflexionMemory]:
        """관련 메모리 검색 (간단한 키워드 매칭)"""
        task_lower = task_description.lower()
        scored_memories = []
        
        for memory in self.memories:
            score = 0
            memory_lower = memory.task_description.lower()
            
            # 키워드 매칭 점수
            task_words = set(task_lower.split())
            memory_words = set(memory_lower.split())
            common_words = task_words & memory_words
            score += len(common_words)
            
            # 실패 메모리에 더 높은 가중치
            if not memory.success:
                score += 2
            
            scored_memories.append((score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored_memories[:top_k]]
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """모든 메모리 반환"""
        return [asdict(memory) for memory in self.memories]


class ReActAgent:
    """ReAct 프레임워크: Reasoning과 Acting 결합"""
    
    def __init__(self, reasoning_bank: ReasoningBank, reflexion: Reflexion):
        self.reasoning_bank = reasoning_bank
        self.reflexion = reflexion
        self.current_steps: List[ReActStep] = []
    
    def think(self, thought: str) -> str:
        """Reasoning 단계: 생각을 정리"""
        return thought
    
    def _generate_answer_from_history(self, task_description: str) -> str:
        """이전 스텝의 검색 결과를 기반으로 구체적인 답변 생성"""
        # 이전 검색 결과 추출
        search_results = []
        for step in self.current_steps:
            if step.action == "search" and "검색 결과:" in step.observation:
                # observation에서 데이터 추출
                observation = step.observation
                if "개의 관련 정보를 찾았습니다" in observation:
                    # 실제 데이터 파싱
                    lines = observation.split("\n")
                    for line in lines[1:]:  # 첫 줄 제외
                        if ": {" in line:
                            search_results.append(line)
        
        # 최근 ReasoningBank 검색 결과 직접 가져오기
        bank_memories = self.reasoning_bank.get_memory_history()
        recent_searches = [m for m in bank_memories if m.get('retrieved_knowledge')]
        
        # 답변 생성
        answer_parts = []
        
        # 질문 타입 분석
        task_lower = task_description.lower()
        
        for memory in recent_searches[-3:]:  # 최근 3개 검색 결과만 사용
            for item in memory.get('retrieved_knowledge', []):
                data = item.get('data', {})
                source = item.get('source', '')
                
                # 질문에 맞는 정보 추출
                if "어디에서 태어났나요" in task_description or "born" in task_lower:
                    if source == 'bio' and 'born_in_city' in data:
                        answer_parts.append(f"{data.get('title', '')}은(는) {data.get('born_in_city', '')}에서 태어났습니다.")
                
                elif "생년" in task_description or "born_year" in task_lower or "언제" in task_description:
                    if source == 'bio' and 'born_year' in data:
                        answer_parts.append(f"{data.get('title', '')}의 생년은 {data.get('born_year', '')}년입니다.")
                
                elif "수도" in task_description or "capital" in task_lower:
                    if source == 'world' and 'capital' in data:
                        answer_parts.append(f"{data.get('title', '')}의 수도는 {data.get('capital', '')}입니다.")
                
                elif "어떤 나라" in task_description or "country" in task_lower:
                    if source == 'corpus' and 'country' in data:
                        answer_parts.append(f"{data.get('title', '')}은(는) {data.get('country', '')}에 있습니다.")
                
                elif "인구" in task_description or "population" in task_lower:
                    if source == 'corpus' and 'population_k' in data:
                        answer_parts.append(f"{data.get('title', '')}의 인구는 약 {data.get('population_k', '')}천 명입니다.")
                
                # 일반적인 경우: 모든 정보 통합
                if not answer_parts and data:
                    info_str = ", ".join([f"{k}: {v}" for k, v in data.items() if k != 'title'])
                    if info_str:
                        answer_parts.append(f"{data.get('title', '')}: {info_str}")
        
        if answer_parts:
            return "\n".join(answer_parts[:3])  # 최대 3개까지만
        else:
            return "검색 결과를 바탕으로 답변을 찾았지만, 구체적인 정보를 추출하지 못했습니다."
    
    def act(self, action: str, action_input: Dict[str, Any]) -> str:
        """Acting 단계: 액션 실행"""
        if action == "search":
            query = action_input.get("query", "")
            results = self.reasoning_bank.search(query)
            if results:
                # 더 상세한 정보 제공
                detail_str = ""
                for r in results[:3]:
                    data = r.get('data', {})
                    source = r.get('source', '')
                    if source == 'bio':
                        detail_str += f"\n[{source}] {data.get('title', '')}: {data.get('bio', '')}\
                            (출생지: {data.get('born_in_city', '')}, 출생년도: {data.get('born_year', '')})"
                    elif source == 'corpus':
                        detail_str += f"\n[{source}] {data.get('title', '')}: {data.get('summary', '')}\
                            (국가: {data.get('country', '')}, 인구: {data.get('population_k', '')}천 명)"
                    elif source == 'world':
                        detail_str += f"\n[{source}] {data.get('title', '')}: 수도 {data.get('capital', '')}, 통화 {data.get('currency', '')}"
                
                return f"검색 결과: {len(results)}개의 관련 정보를 찾았습니다.{detail_str}"
            else:
                return "관련 정보를 찾을 수 없습니다."
        
        elif action == "update":
            key = action_input.get("key", "")
            field = action_input.get("field", "")
            value = action_input.get("value", "")
            source = action_input.get("source", "bio")
            self.reasoning_bank.update(key, field, value, source)
            return f"업데이트 완료: {key}.{field} = {value}"
        
        elif action == "reflect":
            task = action_input.get("task", "")
            memories = self.reflexion.get_relevant_memories(task, top_k=2)
            if memories:
                return f"과거 경험 ({len(memories)}개):\n" + \
                       "\n".join([f"- 에피소드 {m.episode_num}: {m.lesson_learned}" 
                                 for m in memories])
            else:
                return "관련 과거 경험이 없습니다."
        
        elif action == "finish":
            user_answer = action_input.get("answer", "")
            task = action_input.get("task", "")
            
            # 사용자가 답변을 제공했으면 사용, 아니면 자동 생성
            if user_answer and "답변을 찾았습니다" not in user_answer:
                return f"작업 완료: {user_answer}"
            else:
                # 이전 스텝에서 검색 결과를 기반으로 답변 자동 생성
                generated_answer = self._generate_answer_from_history(task if task else "")
                if generated_answer:
                    return f"답변: {generated_answer}"
                else:
                    return f"작업 완료: {user_answer if user_answer else '답변을 찾았습니다.'}"
        
        else:
            return f"알 수 없는 액션: {action}"
    
    def step(self, step_num: int, thought: str, action: str, 
             action_input: Dict[str, Any]) -> Tuple[str, ReActStep]:
        """ReAct 한 단계 실행"""
        # Observation 생성
        observation = self.act(action, action_input)
        
        # Step 저장
        react_step = ReActStep(
            step_num=step_num,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            timestamp=datetime.now().isoformat()
        )
        self.current_steps.append(react_step)
        
        return observation, react_step
    
    def reset(self):
        """현재 스텝 리셋"""
        self.current_steps = []
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """모든 스텝 반환"""
        return [asdict(step) for step in self.current_steps]


class LogManager:
    """모든 로그를 회차별 폴더에 저장하는 관리자"""
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = base_log_dir
        os.makedirs(base_log_dir, exist_ok=True)
    
    def create_episode_dir(self, episode_num: int) -> str:
        """에피소드별 폴더 생성"""
        episode_dir = os.path.join(self.base_log_dir, str(episode_num))
        os.makedirs(episode_dir, exist_ok=True)
        return episode_dir
    
    def save_react_logs(self, episode_dir: str, react_steps: List[Dict[str, Any]]):
        """ReAct reasoning 로그 저장"""
        react_file = os.path.join(episode_dir, "react_reasoning.json")
        with open(react_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_steps': len(react_steps),
                'steps': react_steps
            }, f, ensure_ascii=False, indent=2)
    
    def save_reflexion_logs(self, episode_dir: str, reflexion_memories: List[Dict[str, Any]]):
        """Reflexion 메모리 로그 저장"""
        reflexion_file = os.path.join(episode_dir, "reflexion_memory.json")
        with open(reflexion_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_memories': len(reflexion_memories),
                'memories': reflexion_memories
            }, f, ensure_ascii=False, indent=2)
    
    def save_bank_logs(self, episode_dir: str, bank_memories: List[Dict[str, Any]]):
        """ReasoningBank 메모리 로그 저장"""
        bank_file = os.path.join(episode_dir, "reasoning_bank_memory.json")
        with open(bank_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_memories': len(bank_memories),
                'memories': bank_memories
            }, f, ensure_ascii=False, indent=2)
    
    def save_summary(self, episode_dir: str, episode_num: int, task: str, 
                     success: bool, total_steps: int):
        """에피소드 요약 저장"""
        summary_file = os.path.join(episode_dir, "episode_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'episode_num': episode_num,
                'task': task,
                'success': success,
                'total_steps': total_steps,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)


class ReActReflexionReasoningBank:
    """통합 모델: ReAct + Reflexion + ReasoningBank"""
    
    def __init__(self, bio_path: str, corpus_path: str, world_path: str,
                 log_dir: str = "logs"):
        self.reasoning_bank = ReasoningBank(bio_path, corpus_path, world_path)
        self.reflexion = Reflexion()
        self.react_agent = ReActAgent(self.reasoning_bank, self.reflexion)
        self.log_manager = LogManager(log_dir)
        self.current_episode = 0
    
    def run_episode(self, task_description: str, max_steps: int = 10, 
                   custom_react_prompt: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None) -> bool:
        """한 에피소드 실행"""
        self.current_episode += 1
        episode_num = self.current_episode
        
        print(f"\n{'='*60}")
        print(f"에피소드 {episode_num} 시작: {task_description}")
        print(f"{'='*60}")
        
        # ReAct 에이전트 리셋
        self.react_agent.reset()
        
        # 과거 메모리 검색
        relevant_memories = self.reflexion.get_relevant_memories(task_description)
        if relevant_memories:
            print(f"\n[Reflexion] 관련 과거 경험 {len(relevant_memories)}개 발견:")
            for mem in relevant_memories:
                print(f"  - 에피소드 {mem.episode_num}: {mem.lesson_learned}")
        
        success = False
        step_count = 0
        
        try:
            # 사용자 정의 프롬프트가 있으면 사용, 없으면 자동 생성
            if custom_react_prompt:
                react_sequence = custom_react_prompt
            else:
                # 자동으로 ReAct 시퀀스 생성
                react_sequence = self._generate_react_sequence(task_description)
            
            for step_idx, (thought, action, action_input) in enumerate(react_sequence[:max_steps], 1):
                step_count = step_idx
                
                # finish 액션에 task 정보 추가
                if action == "finish" and "task" not in action_input:
                    action_input["task"] = task_description
                
                observation, react_step = self.react_agent.step(step_idx, thought, action, action_input)
                print(f"\n[Step {step_idx}]")
                print(f"  Thought: {thought}")
                print(f"  Action: {action}")
                print(f"  Observation: {observation[:300]}..." if len(observation) > 300 else f"  Observation: {observation}")
                
                # finish 액션이면 종료
                if action == "finish":
                    success = True
                    break
                
                # 최대 스텝 체크
                if step_count >= max_steps:
                    # 강제로 finish 액션 추가
                    final_thought = "최대 스텝에 도달했으므로 작업을 종료합니다."
                    final_action = "finish"
                    final_input = {"answer": "최대 스텝에 도달하여 작업을 종료했습니다."}
                    self.react_agent.step(step_count + 1, final_thought, final_action, final_input)
                    success = True
                    break
        
        except Exception as e:
            print(f"\n[에러 발생] {str(e)}")
            success = False
        
        # Reflexion에 메모리 추가
        lesson = f"{task_description} 작업에서 {step_count} 스텝을 사용하여 {'성공' if success else '실패'}했습니다."
        self.reflexion.add_memory(
            episode_num=episode_num,
            task_description=task_description,
            success=success,
            failure_reason=None if success else "최대 스텝 초과 또는 에러 발생",
            lesson_learned=lesson
        )
        
        # 로그 저장
        episode_dir = self.log_manager.create_episode_dir(episode_num)
        
        # ReAct 로그 저장
        react_steps = self.react_agent.get_steps()
        self.log_manager.save_react_logs(episode_dir, react_steps)
        
        # Reflexion 로그 저장
        reflexion_memories = self.reflexion.get_all_memories()
        self.log_manager.save_reflexion_logs(episode_dir, reflexion_memories)
        
        # ReasoningBank 로그 저장
        bank_memories = self.reasoning_bank.get_memory_history()
        self.log_manager.save_bank_logs(episode_dir, bank_memories)
        
        # 요약 저장
        self.log_manager.save_summary(episode_dir, episode_num, task_description, 
                                     success, step_count)
        
        print(f"\n[에피소드 {episode_num} 완료] 성공: {success}, 스텝: {step_count}")
        print(f"[로그 저장] {episode_dir}")
        
        return success
    
    def _generate_react_sequence(self, task_description: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """작업 설명을 바탕으로 자동 ReAct 시퀀스 생성"""
        sequence = []
        
        # 키워드 추출
        keywords = task_description.split()
        main_query = keywords[0] if keywords else ""
        
        # Step 1: 검색
        thought1 = f"'{task_description}'라는 질문에 답하기 위해 관련 정보를 지식 베이스에서 검색해야 합니다."
        action1 = "search"
        action_input1 = {"query": main_query}
        sequence.append((thought1, action1, action_input1))
        
        # Step 2: 반영
        thought2 = "과거 유사한 작업 경험을 확인하여 더 나은 접근 방법을 찾겠습니다."
        action2 = "reflect"
        action_input2 = {"task": task_description}
        sequence.append((thought2, action2, action_input2))
        
        # Step 3: 완료 (답변 자동 생성)
        thought3 = "검색 결과와 과거 경험을 바탕으로 답변을 정리했습니다."
        action3 = "finish"
        action_input3 = {"answer": "", "task": task_description}  # 빈 answer로 설정하여 자동 생성 유도
        sequence.append((thought3, action3, action_input3))
        
        return sequence


def main():
    """메인 실행 함수"""
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
    
    # 여러 에피소드 실행
    tasks = [
        "Liam Brook은 어디에서 태어났나요?",
        "Ava Shin의 생년은 언제인가요?",
        "Norland의 수도는 어디인가요?",
        "Seoria는 어떤 나라에 있나요?"
    ]
    
    for task in tasks:
        model.run_episode(task, max_steps=10)
        print("\n")


if __name__ == "__main__":
    main()

