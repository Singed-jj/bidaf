### 환경
- 1. python.__version__ : 3.7.1
- 2. tensorflow.__version__ : 1.14

### 구현정도
- model : modeling layer 에 "ValueError: Dimensions must be equal, but are 300 and 900 for 'model_cpu0/modeling_layer/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_1' (op: 'MatMul') with input shapes: [3600,300], [900,400]." 버그 존재.
- trainer : 구현 완료.
- f1, EM score : 구현 못함.

### 구현순서
- 1. model/pre_processor.py
- 2. model/bidaf_model.py
- 3. train/bidaf_trainer.py
- 4. main/main.py

### 전처리로 만들어준 dataset
- 1. "data/squad/data_train.json"
- 2. "data/squad/data_dev.json"
- 3. "data/squad/data_test.json"
- 4. "data/squad/fixed_data_train.json"
- 5. "data/squad/fixed_data_dev.json"
- 6. "data/squad/fixed_data_test.json"
- 7. "data/squad/data_total.json"
- 8. "data/squad/fixed_data_total.json"

### 학습방법
- 루트 디렉토리에서
- > python models/pre_processor.py
- > python main/main.py "configs/config.json" --mode=train

### bidaf_trainer
- lookup : batch 에서 word -> index, char -> index 로 바꿔 행렬 연산이 가능하게 만들어준다.
- train_step : 매 train step 마다 실행되는 함수이며 model 에 훈련에 필요한 [x, label] 쌍 을 피딩한다. dropout 은 0.2 로 두었다.
- dev_step : train 중 model 의 정확도를 보려는 optional step 이며, dropout 1.0 으로 두어 실제 model 을 학습시키지는 않는다.

### bidaf_model
- character_embedding_layer : filter size(7,7,3,3,3,3) 을 각각 가진 6개의 convolutionl layer 와 3개의 fully connected layer 로 구성.

- contextual_embedding_layer :
  - word embedding : 300dimensions 을 가진 word2vec 모델을 사용하여 임베딩
  - char embedding : character_embedding_layer 에서 임베딩
  - word embedding 과 char embedding 을 concat 하여 양방향 LSTM 에 넣어 hidden_state output 을 다음 attention layer 의 input 으로 사용

- attention_embedding_layer : tf.expand_dims, tf.tile 함수를 사용하여 쉽게 행렬 elementwise multiplication

- modeling_layer : 2번의 양방향 LSTM output 과 그것을 한번 더 양방향 LSTM 에 넣어 나온 output 을 리턴.

- output_layer : 마지막으로 훈련가능한 variable weight 를 만들고, model loss 계산을 위한 최종 output 를 softmax 함수에 넣어준다.

- build_model : 위의 layer 5 개를 연결하여 모델을 빌드

- build_loss : 마지막 layer 인 output_layer 에서 구한 softmax 값과 expected value 와의 cross-entropy 함수를 이용하여 loss 함수 구성.
