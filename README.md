# SBERT_test
2022/10/24 SBERT_beta
```
models : 모델 구현 코드.
datasets : pytorch dataset 구현 코드.
```

# 전처리 방법
sbert_config.yaml에서 경로 지정한 뒤에,
```
python preprocess.py
```
실행

# 학습 방법

```
wandbd login
```
으로 wandb login 먼저 한다.

sbert_config.yaml에서 config 수정 뒤에,
```
python train.py
```
로 실행.
