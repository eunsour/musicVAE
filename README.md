# A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music
기존 VAE는 의미 있는 잠재 표현을 생성하는 효과적인 모델이었지만, 순차 데이터에 대한 적용이 제한적이었고, 장기 구조를 가진 시퀀스를 모델링하는 데 제한적이었다.  
따라서 음악 구조를 반영할 수 있는 latent space를 Recurrent VAE 기반으로 Hierarchical Decoder를 추가한 MusicVAE를 제안한다.
- Encoder로 2 Layer-Bidirectional LSTM 사용
- Decoder에 Conductor를 추가하여 Hierarchical하게 구성
- Conductor는 Latent Vector Z를 입력받아, U차원으로 임베딩
- Decoder는 U개의 Vector를 받아 결과 값 출력
- 긴 시퀀스에 대한 posterier collapse 문제 해결
- Latent space interpolation 가능

<br>

## Preprocessing
1. 4-4 박자인 .mid 파일들만 추출
2. 드럼인지 확인
3. 3차원 배치로 변환
4. 원 핫 인코딩
5. pickle 파일로 저장

<br>

## Training
- Encoder : 2 Layer-Bidirectional LSTM
- Decoder : 2 Layer-Unidirectional LSTM 
- Conductor : 2 Layer-Unidirectional LSTM
- models 폴더에 checkpoint 파일 저장

<br>

## Generation
- make_gererating_model 함수에서 생성된 임의의 latent_vector에서 one_hot 출력 생성
- onehot_idx -> drum_seq -> 드럼 채널 클래스 가져오기
- generated 폴더에 드럼 파일(.midi) 생성

<br>

## 실행
1. Installation
- `pip install -r requirements.txt`
- Install **torch** for your version

2. Preprocessing Data
- `python3 preprocess.py`

3. Model Training
- `python3 train.py`

4. Generator 
- `python3 generator.py`
