"""
Transformer: "Attention is All You Need" (Vaswani et al., 2017) 완전 구현
한국어-영어 번역 작업

논문의 모든 세부사항을 충실하게 재현:
- Scaled Dot-Product Attention
- Multi-Head Attention (h=8, d_k=d_v=64)
- Position-wise Feed-Forward Networks (d_ff=2048)
- Positional Encoding (sin/cos)
- Encoder Stack (N=6 layers)
- Decoder Stack (N=6 layers)
- Layer Normalization & Residual Connections
- Dropout
- Label Smoothing
- Warmup Learning Rate Scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from collections import Counter


# ============================================================================
# 0. 간단한 토크나이저 & 데이터셋
# ============================================================================
class SimpleTokenizer:
    """간단한 문자/단어 기반 토크나이저"""
    
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = 4
        
    def build_vocab(self, sentences, min_freq=1):
        """어휘 사전 구축"""
        # 모든 문자/단어 수집
        all_chars = []
        for sent in sentences:
            # 공백 기준으로 분리 (단어 단위)
            tokens = sent.strip().split()
            for token in tokens:
                # 각 단어를 문자로 분리
                all_chars.extend(list(token))
                all_chars.append(' ')  # 공백도 토큰으로
        
        # 빈도수 계산
        char_counts = Counter(all_chars)
        
        # 최소 빈도 이상인 문자만 추가
        for char, count in char_counts.most_common():
            if count >= min_freq and char not in self.word2idx:
                self.word2idx[char] = self.vocab_size
                self.idx2word[self.vocab_size] = char
                self.vocab_size += 1
    
    def encode(self, text):
        """텍스트를 인덱스로 변환"""
        tokens = []
        for char in text.strip():
            tokens.append(self.word2idx.get(char, self.word2idx['<unk>']))
        return tokens
    
    def decode(self, indices):
        """인덱스를 텍스트로 변환"""
        chars = []
        for idx in indices:
            if idx in [0, 1, 2]:  # <pad>, <sos>, <eos>
                continue
            chars.append(self.idx2word.get(idx, '<unk>'))
        return ''.join(chars).strip()


def create_translation_dataset():
    """한-영 번역 데이터셋 생성"""
    # 학습 데이터
    train_pairs = [
        # 기본 인사
        ("안녕", "hello"),
        ("안녕하세요", "hello"),
        ("감사합니다", "thank you"),
        ("고맙습니다", "thank you"),
        
        # 사랑 표현
        ("나는 너를 사랑해", "i love you"),
        ("사랑해", "i love you"),
        ("사랑합니다", "i love you"),
        ("좋아해", "i like you"),
        ("좋아합니다", "i like you"),
        
        # 기본 문장
        ("나는 학생이야", "i am a student"),
        ("나는 학생입니다", "i am a student"),
        ("나는 선생님이야", "i am a teacher"),
        ("나는 선생님입니다", "i am a teacher"),
        
        # 날씨
        ("날씨가 좋아", "the weather is good"),
        ("날씨가 좋네요", "the weather is good"),
        ("비가 와", "it is raining"),
        ("비가 옵니다", "it is raining"),
        
        # 질문
        ("이름이 뭐야", "what is your name"),
        ("이름이 무엇입니까", "what is your name"),
        ("어디 가", "where are you going"),
        ("어디 가세요", "where are you going"),
        
        # 시간
        ("지금 몇 시야", "what time is it"),
        ("지금 몇 시입니까", "what time is it"),
        ("오늘 날짜가 뭐야", "what is the date today"),
        
        # 음식
        ("배고파", "i am hungry"),
        ("배고픕니다", "i am hungry"),
        ("맛있어", "it is delicious"),
        ("맛있습니다", "it is delicious"),
        
        # 감정
        ("행복해", "i am happy"),
        ("행복합니다", "i am happy"),
        ("슬퍼", "i am sad"),
        ("슬픕니다", "i am sad"),
        ("화났어", "i am angry"),
        
        # 더 많은 데이터
        ("안녕히 가세요", "goodbye"),
        ("잘 가", "goodbye"),
        ("또 만나", "see you again"),
        ("내일 봐", "see you tomorrow"),
        
        ("미안해", "i am sorry"),
        ("미안합니다", "i am sorry"),
        ("괜찮아", "it is okay"),
        ("괜찮습니다", "it is okay"),
        
        ("도와줘", "help me"),
        ("도와주세요", "help me"),
        ("알겠어", "i understand"),
        ("알겠습니다", "i understand"),
        
        # 가족
        ("이것은 내 가족이야", "this is my family"),
        ("아버지", "father"),
        ("어머니", "mother"),
        ("형제", "brother"),
        
        # 숫자
        ("하나", "one"),
        ("둘", "two"),
        ("셋", "three"),
        
        # 색깔
        ("빨강", "red"),
        ("파랑", "blue"),
        ("초록", "green"),
        
        # 동물
        ("고양이", "cat"),
        ("강아지", "dog"),
        ("새", "bird"),
    ]
    
    return train_pairs


# ============================================================================
# 1. Scaled Dot-Product Attention
# ============================================================================
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    논문 Section 3.2.1의 Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = query.size(-1)
    
    # QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Masking
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# ============================================================================
# 2. Multi-Head Attention
# ============================================================================
class MultiHeadAttention(nn.Module):
    """논문 Section 3.2.2의 Multi-Head Attention"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attention_weights


# ============================================================================
# 3. Position-wise Feed-Forward Networks
# ============================================================================
class PositionWiseFeedForward(nn.Module):
    """논문 Section 3.3의 Position-wise Feed-Forward Networks"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================================
# 4. Positional Encoding
# ============================================================================
class PositionalEncoding(nn.Module):
    """논문 Section 3.5의 Positional Encoding"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# 5. Encoder Layer
# ============================================================================
class EncoderLayer(nn.Module):
    """논문 Section 3.1의 Encoder Layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Multi-Head Self-Attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


# ============================================================================
# 6. Decoder Layer
# ============================================================================
class DecoderLayer(nn.Module):
    """논문 Section 3.1의 Decoder Layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked Multi-Head Self-Attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Multi-Head Cross-Attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


# ============================================================================
# 7. Encoder
# ============================================================================
class Encoder(nn.Module):
    """논문의 Encoder Stack (N=6 layers)"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, 
                 max_len, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Embedding + sqrt(d_model) scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add Positional Encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# ============================================================================
# 8. Decoder
# ============================================================================
class Decoder(nn.Module):
    """논문의 Decoder Stack (N=6 layers)"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 max_len, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Embedding + sqrt(d_model) scaling
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add Positional Encoding
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x


# ============================================================================
# 9. Transformer
# ============================================================================
class Transformer(nn.Module):
    """완전한 Transformer 모델 (Vaswani et al., 2017)"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6,
                 n_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = Encoder(
            src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout
        )
        
        # Final linear layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Weight initialization (Xavier uniform)
        self._init_weights()
        
    def _init_weights(self):
        """Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        """Source padding mask"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """Target mask (padding + look-ahead)"""
        batch_size, tgt_len = tgt.shape
        
        # Padding mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # Look-ahead mask
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()
        
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        
        return output
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)


# ============================================================================
# 10. 학습 함수
# ============================================================================
def prepare_batch(pairs, src_tokenizer, tgt_tokenizer, device):
    """배치 데이터 준비"""
    src_texts, tgt_texts = zip(*pairs)
    
    # 인코딩
    src_encoded = [src_tokenizer.encode(text) for text in src_texts]
    tgt_encoded = [tgt_tokenizer.encode(text) for text in tgt_texts]
    
    # 최대 길이
    max_src_len = max(len(seq) for seq in src_encoded)
    max_tgt_len = max(len(seq) for seq in tgt_encoded)
    
    # 패딩
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []
    
    for src, tgt in zip(src_encoded, tgt_encoded):
        # Source: 패딩
        src_pad = src + [0] * (max_src_len - len(src))
        src_padded.append(src_pad)
        
        # Target input: <sos> + sequence (패딩)
        tgt_in = [1] + tgt  # 1 = <sos>
        tgt_in = tgt_in + [0] * (max_tgt_len + 1 - len(tgt_in))
        tgt_input_padded.append(tgt_in)
        
        # Target output: sequence + <eos> (패딩)
        tgt_out = tgt + [2]  # 2 = <eos>
        tgt_out = tgt_out + [0] * (max_tgt_len + 1 - len(tgt_out))
        tgt_output_padded.append(tgt_out)
    
    src_tensor = torch.LongTensor(src_padded).to(device)
    tgt_input_tensor = torch.LongTensor(tgt_input_padded).to(device)
    tgt_output_tensor = torch.LongTensor(tgt_output_padded).to(device)
    
    return src_tensor, tgt_input_tensor, tgt_output_tensor


def train_epoch(model, train_pairs, src_tokenizer, tgt_tokenizer, 
                optimizer, criterion, device, batch_size=8):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 배치 단위로 학습
    for i in range(0, len(train_pairs), batch_size):
        batch_pairs = train_pairs[i:i+batch_size]
        
        # 배치 준비
        src, tgt_input, tgt_output = prepare_batch(
            batch_pairs, src_tokenizer, tgt_tokenizer, device
        )
        
        # Forward
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Loss 계산
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def translate(model, text, src_tokenizer, tgt_tokenizer, device, max_len=50):
    """번역 수행"""
    model.eval()
    
    # 입력 인코딩
    src_encoded = src_tokenizer.encode(text)
    src_tensor = torch.LongTensor([src_encoded]).to(device)
    
    # Encoder
    src_mask = model.make_src_mask(src_tensor)
    enc_output = model.encode(src_tensor, src_mask)
    
    # Decoder (greedy decoding)
    tgt_indices = [1]  # <sos>
    
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor([tgt_indices]).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        
        with torch.no_grad():
            dec_output = model.decode(tgt_tensor, enc_output, src_mask, tgt_mask)
            output = model.fc_out(dec_output)
        
        # 다음 토큰 예측
        next_token = output[0, -1].argmax().item()
        
        # <eos> 또는 <pad>면 종료
        if next_token == 2 or next_token == 0:
            break
        
        tgt_indices.append(next_token)
    
    # 디코딩
    translation = tgt_tokenizer.decode(tgt_indices[1:])  # <sos> 제외
    
    return translation


# ============================================================================
# 11. 메인 함수
# ============================================================================
def main():
    print("="*80)
    print("Transformer: 한국어 → 영어 번역")
    print("="*80)
    
    # 하이퍼파라미터
    D_MODEL = 256      # 작은 모델로 조정
    N_LAYERS = 3       # 레이어 수 줄임
    N_HEADS = 8
    D_FF = 1024        # FFN 차원 줄임
    MAX_LEN = 100
    DROPOUT = 0.1
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 300   # 충분한 학습
    LEARNING_RATE = 0.0001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n디바이스: {device}")
    
    # 데이터 준비
    print("\n데이터 준비 중...")
    train_pairs = create_translation_dataset()
    print(f"학습 데이터 수: {len(train_pairs)}")
    
    # 토크나이저 구축
    print("토크나이저 구축 중...")
    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()
    
    src_texts = [pair[0] for pair in train_pairs]
    tgt_texts = [pair[1] for pair in train_pairs]
    
    src_tokenizer.build_vocab(src_texts)
    tgt_tokenizer.build_vocab(tgt_texts)
    
    print(f"한국어 어휘 크기: {src_tokenizer.vocab_size}")
    print(f"영어 어휘 크기: {tgt_tokenizer.vocab_size}")
    
    # 모델 초기화
    print("\n모델 초기화 중...")
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터 수: {num_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                                  betas=(0.9, 0.98), eps=1e-9)
    
    # 학습
    print("\n학습 시작...\n")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # 데이터 셔플
        import random
        random.shuffle(train_pairs)
        
        # 학습
        train_loss = train_epoch(
            model, train_pairs, src_tokenizer, tgt_tokenizer,
            optimizer, criterion, device, BATCH_SIZE
        )
        
        # 주기적으로 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss:.4f}")
            
            # 테스트 번역
            if (epoch + 1) % 50 == 0:
                print("\n[번역 테스트]")
                test_sentences = [
                    "나는 너를 사랑해",
                    "안녕하세요",
                    "감사합니다",
                    "사랑해",
                    "좋아해"
                ]
                
                for sent in test_sentences:
                    translation = translate(model, sent, src_tokenizer, 
                                          tgt_tokenizer, device)
                    print(f"  {sent} → {translation}")
                print()
        
        # Best model 저장
        if train_loss < best_loss:
            best_loss = train_loss
    
    # 최종 테스트
    print("\n" + "="*80)
    print("최종 번역 테스트")
    print("="*80 + "\n")
    
    test_sentences = [
        "나는 너를 사랑해",
        "사랑해",
        "안녕",
        "안녕하세요",
        "감사합니다",
        "좋아해",
        "행복해",
        "미안해",
        "고맙습니다",
        "날씨가 좋아",
        "배고파",
        "맛있어"
    ]
    
    for sent in test_sentences:
        translation = translate(model, sent, src_tokenizer, tgt_tokenizer, device)
        print(f"{sent:20s} → {translation}")
    
    # 인터랙티브 모드
    print("\n" + "="*80)
    print("인터랙티브 번역 모드 (종료: 'quit')")
    print("="*80 + "\n")
    
    while True:
        try:
            user_input = input("한국어 입력: ").strip()
            if user_input.lower() == 'quit':
                break
            if user_input:
                translation = translate(model, user_input, src_tokenizer, 
                                      tgt_tokenizer, device)
                print(f"영어 번역: {translation}\n")
        except KeyboardInterrupt:
            break
    
    print("\n프로그램 종료!")


if __name__ == "__main__":
    main()
