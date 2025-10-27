import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import List, Dict, Tuple
import random

# 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {DEVICE}")

# 하이퍼파라미터 (GPT-2 스타일)
class Config:
    def __init__(self):
        self.vocab_size = 10000  # 어휘 크기
        self.d_model = 768      # 모델 차원 (GPT-2는 더 큼)
        self.n_heads = 12       # 어텐션 헤드 수 (GPT-2는 더 많음)
        self.n_layers = 12      # 레이어 수 (GPT-2는 더 많음)
        self.d_ff = 3072       # 피드포워드 차원 (4 * d_model)
        self.max_seq_len = 1024 # 최대 시퀀스 길이 (GPT-2는 더 김)
        self.dropout = 0.1      # 드롭아웃 비율
        self.lr = 0.0001        # 학습률
        self.batch_size = 16    # 배치 크기 (더 큰 모델이므로 작게)
        self.epochs = 100       # 에포크 수

config = Config()

# GPT-2 스타일 토크나이저
class GPT2Tokenizer:
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts: List[str]):
        """텍스트에서 어휘 구축"""
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도가 높은 단어들을 어휘에 추가
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words:
            if word not in self.word_to_idx and len(self.word_to_idx) < config.vocab_size:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        print(f"어휘 크기: {self.vocab_size}")
    
    def encode(self, text: str) -> List[int]:
        """텍스트를 인덱스로 변환"""
        words = text.split()
        indices = []
        for word in words:
            idx = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
            indices.append(idx)
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """인덱스를 텍스트로 변환"""
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in ['<PAD>', '<SOS>', '<EOS>']:
                    words.append(word)
        return ' '.join(words)

class GPT2PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(positions)
        return pos_embeddings

# GPT-2 스타일의 Multi-Head Attention (개선된 어텐션)
class GPT2MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.c_attn = nn.Linear(d_model, 3 * d_model)  # Q, K, V를 한번에
        self.c_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Q, K, V 계산
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # 헤드로 분할
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 어텐션 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 어텐션 적용
        attention_output = torch.matmul(attention_weights, v)
        
        # 헤드 연결
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        # 최종 선형 변환
        output = self.c_proj(attention_output)
        
        return output, attention_weights

# GPT-2 스타일의 Transformer Block (Pre-LN 구조)
class GPT2Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = GPT2MultiHeadAttention(d_model, n_heads, dropout)
        
        # GPT-2 스타일 피드포워드 (GELU 활성화 함수 사용)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT-2는 GELU 사용
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask=None):
        # Pre-LN 구조 (GPT-2 스타일)
        # Self-attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, causal_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        norm_x = self.ln2(x)
        ff_out = self.mlp(norm_x)
        x = x + self.dropout(ff_out)
        
        return x

# GPT-2 모델
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)  # Word Token Embedding
        self.wpe = GPT2PositionalEncoding(config.d_model, config.max_seq_len)  # Word Position Embedding
        self.h = nn.ModuleList([
            GPT2Block(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len):
        """Causal mask 생성 (GPT 스타일)"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(DEVICE)
    
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        x = self.wte(input_ids) + self.wpe(input_ids)
        causal_mask = self.create_causal_mask(seq_len)
        for block in self.h:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

# 번역 데이터셋 생성
def create_translation_dataset():
    """한국어-영어 번역 데이터셋 생성"""
    korean_sentences = [
        "나는 밥을 먹는다",
        "나는 물을 마신다", 
        "나는 책을 읽는다",
        "나는 학교에 간다",
        "나는 친구를 만난다",
        "나는 집에 온다",
        "나는 음악을 듣는다",
        "나는 영화를 본다",
        "나는 운동을 한다",
        "나는 잠을 잔다",
        "나는 공부를 한다",
        "나는 게임을 한다",
        "나는 요리를 한다",
        "나는 쇼핑을 한다",
        "나는 여행을 한다"
    ]
    
    english_sentences = [
        "I have breakfast",
        "I drink water",
        "I read a book", 
        "I go to school",
        "I meet my friend",
        "I come home",
        "I listen to music",
        "I watch a movie",
        "I exercise",
        "I sleep",
        "I study",
        "I play games",
        "I cook",
        "I go shopping",
        "I travel"
    ]
    
    return korean_sentences, english_sentences

# 데이터 로더
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, korean_texts, english_texts, korean_tokenizer, english_tokenizer):
        self.korean_texts = korean_texts
        self.english_texts = english_texts
        self.korean_tokenizer = korean_tokenizer
        self.english_tokenizer = english_tokenizer
    
    def __len__(self):
        return len(self.korean_texts)
    
    def __getitem__(self, idx):
        korean_text = self.korean_texts[idx]
        english_text = self.english_texts[idx]
        
        # 한국어 입력 (SOS 토큰 추가)
        korean_tokens = [2] + self.korean_tokenizer.encode(korean_text) + [3]  # SOS + tokens + EOS
        
        # 영어 타겟 (SOS 토큰 추가)
        english_tokens = [2] + self.english_tokenizer.encode(english_text) + [3]  # SOS + tokens + EOS
        
        # 패딩
        max_len = config.max_seq_len
        korean_tokens = korean_tokens[:max_len]
        english_tokens = english_tokens[:max_len]
        
        korean_tokens += [0] * (max_len - len(korean_tokens))
        english_tokens += [0] * (max_len - len(english_tokens))
        
        return torch.tensor(korean_tokens, dtype=torch.long), torch.tensor(english_tokens, dtype=torch.long)

# 훈련 함수
def train_model(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (korean_input, english_target) in enumerate(dataloader):
        korean_input = korean_input.to(DEVICE)
        english_target = english_target.to(DEVICE)
        
        optimizer.zero_grad()
        
        # 모델 출력
        logits = model(korean_input)
        
        # 손실 계산 (teacher forcing)
        loss = criterion(logits.view(-1, logits.size(-1)), english_target.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 추론 함수
def translate(model, korean_text, korean_tokenizer, english_tokenizer):
    model.eval()
    
    # 한국어 텍스트 토크나이징
    korean_tokens = [2] + korean_tokenizer.encode(korean_text) + [3]  # SOS + tokens + EOS
    
    # 패딩
    max_len = config.max_seq_len
    korean_tokens = korean_tokens[:max_len]
    korean_tokens += [0] * (max_len - len(korean_tokens))
    
    input_tensor = torch.tensor([korean_tokens], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_tensor)
        
        # 가장 높은 확률의 토큰들 선택
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        # 영어 토큰을 텍스트로 변환
        english_tokens = predicted_tokens[0].cpu().tolist()
        english_text = english_tokenizer.decode(english_tokens)
        
        return english_text

# 메인 실행 함수
def main():
    print("GPT-2 기반 한국어-영어 번역 모델 시작")
    
    # 데이터셋 생성
    korean_texts, english_texts = create_translation_dataset()
    
    # 토크나이저 생성
    korean_tokenizer = GPT2Tokenizer()
    english_tokenizer = GPT2Tokenizer()
    
    # 어휘 구축
    korean_tokenizer.build_vocab(korean_texts)
    english_tokenizer.build_vocab(english_texts)
    
    # 데이터셋 및 데이터로더
    dataset = TranslationDataset(korean_texts, english_texts, korean_tokenizer, english_tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 모델 생성
    model = GPT2(config).to(DEVICE)
    
    # 옵티마이저 및 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD 토큰 무시
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련
    print("훈련 시작...")
    for epoch in range(config.epochs):
        avg_loss = train_model(model, dataloader, optimizer, criterion, epoch)
        
        # 0, 20, 40, 60, 80, 100 에포크마다 번역 테스트
        if epoch % 20 == 0 or epoch == 0:
            test_text = "나는 밥을 먹는다"
            translation = translate(model, test_text, korean_tokenizer, english_tokenizer)
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            print(f"테스트 번역: '{test_text}' -> '{translation}'")
            print("-" * 50)
    
    # 최종 테스트
    print("\n최종 번역 테스트:")
    test_texts = ["나는 밥을 먹는다", "나는 물을 마신다", "나는 책을 읽는다"]
    
    for test_text in test_texts:
        translation = translate(model, test_text, korean_tokenizer, english_tokenizer)
        print(f"'{test_text}' -> '{translation}'")
    
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'korean_tokenizer': korean_tokenizer,
        'english_tokenizer': english_tokenizer,
        'config': config
    }, 'gpt2_translation_model.pth')
    
    print("모델이 'gpt2_translation_model.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()
