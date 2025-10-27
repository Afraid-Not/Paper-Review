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

# BERT 원본 논문 하이퍼파라미터
class BertConfig:
    def __init__(self):
        # BERT-Base 설정
        self.vocab_size = 30000      # WordPiece 어휘 크기
        self.hidden_size = 768        # BERT-Base 차원
        self.num_hidden_layers = 12   # BERT-Base 레이어 수
        self.num_attention_heads = 12 # BERT-Base 어텐션 헤드 수
        self.intermediate_size = 3072 # 피드포워드 차원 (4 * hidden_size)
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512  # 최대 시퀀스 길이
        self.type_vocab_size = 2     # 세그먼트 타입 (문장 A, B)
        self.initializer_range = 0.02
        
        # 훈련 설정
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_train_epochs = 3
        self.warmup_steps = 10000
        self.max_steps = 100000

config = BertConfig()

# BERT 스타일 토크나이저 (WordPiece 기반)
class BertTokenizer:
    def __init__(self):
        # BERT 특수 토큰들
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.vocab_size = 5
        
    def build_vocab(self, texts: List[str]):
        """텍스트에서 WordPiece 어휘 구축"""
        word_freq = {}
        for text in texts:
            # 간단한 단어 분할 (실제로는 WordPiece 알고리즘 사용)
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도가 높은 단어들을 어휘에 추가
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words:
            if word not in self.vocab and len(self.vocab) < config.vocab_size:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.ids_to_tokens[idx] = word
        
        self.vocab_size = len(self.vocab)
        print(f"BERT 어휘 크기: {self.vocab_size}")
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할"""
        return text.split()
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """토큰을 ID로 변환"""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab['[UNK]']))
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """ID를 토큰으로 변환"""
        tokens = []
        for id in ids:
            tokens.append(self.ids_to_tokens.get(id, '[UNK]'))
        return tokens

# BERT 임베딩 레이어
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 임베딩 합산
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# BERT 어텐션 레이어
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({config.hidden_size})는 num_attention_heads ({config.num_attention_heads})로 나누어떨어져야 합니다")
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 어텐션 점수 계산
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 정규화
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # 어텐션 적용
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

# BERT 어텐션 출력 레이어
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# BERT 어텐션 레이어 (SelfAttention + SelfOutput)
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        
    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

# BERT 중간 레이어 (피드포워드)
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()  # BERT는 GELU 사용
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# BERT 출력 레이어
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# BERT 레이어 (Attention + Intermediate + Output)
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_outputs)
        layer_output = self.output(intermediate_output, attention_outputs)
        return layer_output

# BERT 인코더
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

# BERT 풀러 (CLS 토큰에서 특징 추출)
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # CLS 토큰의 출력 사용
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 어텐션 마스크 확장
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs)
        
        return encoder_outputs, pooled_output

# BERT 사전훈련 헤드들
class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        # 가중치 공유
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        
    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

# BERT 사전훈련 모델 (MLM + NSP)
class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.get_input_embeddings().weight)
        self.next_sentence = BertOnlyNSPHead(config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, labels=None, next_sentence_label=None):
        
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        sequence_output, pooled_output = outputs
        
        prediction_scores = self.cls(sequence_output)
        seq_relationship_score = self.next_sentence(pooled_output)
        
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
        
        return {
            'loss': total_loss,
            'prediction_logits': prediction_scores,
            'seq_relationship_logits': seq_relationship_score,
            'hidden_states': sequence_output,
            'attentions': None
        }

# BERT 분류 모델 (다운스트림 태스크용)
class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, labels=None):
        
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs[0],
            'attentions': None
        }

# BERT 번역 모델
class BertForTranslation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.translation_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, labels=None):
        
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        translation_logits = self.translation_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(translation_logits.view(-1, self.translation_head.out_features), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': translation_logits,
            'hidden_states': sequence_output,
            'attentions': None
        }

# 데이터셋 생성 (BERT 스타일)
def create_bert_dataset():
    """BERT 사전훈련용 데이터셋 생성"""
    # 문장 쌍 데이터 (NSP 태스크용)
    sentence_pairs = [
        ("나는 밥을 먹는다", "I have breakfast"),
        ("나는 물을 마신다", "I drink water"),
        ("나는 책을 읽는다", "I read a book"),
        ("나는 학교에 간다", "I go to school"),
        ("나는 친구를 만난다", "I meet my friend"),
        ("나는 집에 온다", "I come home"),
        ("나는 음악을 듣는다", "I listen to music"),
        ("나는 영화를 본다", "I watch a movie"),
        ("나는 운동을 한다", "I exercise"),
        ("나는 잠을 잔다", "I sleep"),
    ]
    
    return sentence_pairs

# 간단한 번역 데이터셋 생성
def create_simple_translation_dataset():
    """간단한 번역 데이터셋"""
    translations = {
        "나는 밥을 먹는다": "I have breakfast",
        "나는 물을 마신다": "I drink water", 
        "나는 책을 읽는다": "I read a book",
        "나는 학교에 간다": "I go to school",
        "나는 친구를 만난다": "I meet my friend",
        "나는 집에 온다": "I come home",
        "나는 음악을 듣는다": "I listen to music",
        "나는 영화를 본다": "I watch a movie",
        "나는 운동을 한다": "I exercise",
        "나는 잠을 잔다": "I sleep"
    }
    return translations

# 간단한 BERT 입력 생성
def create_simple_bert_input(korean_text, tokenizer, max_length=64):
    """간단한 BERT 입력 생성"""
    
    # 토큰화
    korean_tokens = tokenizer.tokenize(korean_text)
    
    # BERT 입력 형식: [CLS] 한국어 [SEP]
    tokens = ['[CLS]'] + korean_tokens + ['[SEP]']
    
    # 길이 제한
    if len(tokens) > max_length:
        tokens = tokens[:max_length-1] + ['[SEP]']
    
    # ID 변환
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 패딩
    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.vocab['[PAD]']] * padding_length
    
    # 어텐션 마스크
    attention_mask = [1] * len(tokens) + [0] * padding_length
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }

# 간단한 훈련 함수 (분류 태스크로 번역)
def train_bert_simple(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # BERT 분류 모델 출력
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 간단한 BERT 추론 함수 (분류 결과를 번역으로 변환)
def bert_translate_simple(model, korean_text, tokenizer, translation_dict):
    model.eval()
    
    # 입력 생성
    inputs = create_simple_bert_input(korean_text, tokenizer)
    
    input_ids = inputs['input_ids'].unsqueeze(0).to(DEVICE)
    attention_mask = inputs['attention_mask'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        
        # 분류 결과
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        # 클래스에 따른 번역 반환
        if predicted_class < len(translation_dict):
            translations = list(translation_dict.values())
            return translations[predicted_class]
        else:
            return "Unknown translation"

# 메인 실행 함수
def main():
    print("BERT 원본 논문 구조 기반 모델 시작")
    
    # 간단한 번역 데이터셋 생성
    translation_dict = create_simple_translation_dataset()
    
    # 토크나이저 생성
    tokenizer = BertTokenizer()
    
    # 어휘 구축
    all_texts = list(translation_dict.keys()) + list(translation_dict.values())
    tokenizer.build_vocab(all_texts)
    
    # 모델 생성 (분류 모델로 번역 태스크 수행)
    num_classes = len(translation_dict)
    model = BertForSequenceClassification(config, num_labels=num_classes).to(DEVICE)
    
    # 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"BERT 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 간단한 데이터로더 생성
    dataset = []
    for i, (korean_text, english_text) in enumerate(translation_dict.items()):
        inputs = create_simple_bert_input(korean_text, tokenizer)
        inputs['labels'] = torch.tensor(i, dtype=torch.long)  # 클래스 라벨
        dataset.append(inputs)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 훈련
    print("BERT 번역 훈련 시작...")
    for epoch in range(config.num_train_epochs):
        avg_loss = train_bert_simple(model, dataloader, optimizer, epoch)
        
        # 0, 1, 2 에포크마다 테스트
        if epoch % 1 == 0 or epoch == 0:
            test_text = "나는 밥을 먹는다"
            translation = bert_translate_simple(model, test_text, tokenizer, translation_dict)
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            print(f"BERT 번역 결과: '{test_text}' -> '{translation}'")
            print("-" * 50)
    
    # 최종 테스트
    print("\n최종 BERT 번역 테스트:")
    test_texts = ["나는 밥을 먹는다", "나는 물을 마신다", "나는 책을 읽는다"]
    
    for test_text in test_texts:
        translation = bert_translate_simple(model, test_text, tokenizer, translation_dict)
        print(f"'{test_text}' -> '{translation}'")
    
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': config,
        'translation_dict': translation_dict
    }, 'bert_paper_model.pth')
    
    print("BERT 모델이 'bert_paper_model.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()