### Rank별 train/loss
https://api.wandb.ai/links/jhtwiz00/0ybl8him
<img width="1696" alt="image" src="https://github.com/user-attachments/assets/f0dc393d-1c48-4421-b7bd-f4ac707fd429">

### Rank별 runtime
https://api.wandb.ai/links/jhtwiz00/uvcatxmr
<img width="1691" alt="image" src="https://github.com/user-attachments/assets/c65eae10-c4da-4175-8271-b027174d395f">


### Rank별 메모리 점유
#### print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')로 찍은 기준
- Rank 8: 3.9 GB
- Rank 128: 4.6 GB
- Rank 256: 5.3 GB
#### wandb 기준
https://api.wandb.ai/links/jhtwiz00/lkq8udzd
<img width="1695" alt="image" src="https://github.com/user-attachments/assets/346fdf9f-4821-48ab-8cd7-11dfd552bfad">
wandb와 print로 찍은 값이 다르다.

### 총 정리
해당 모델에서 input_features와 output_features가 1024 혹은 512 였는데 Lora 적용 후 8/128/256 feature로 계산되다보니 **rank가 낮을수록 속도가 빠르고 GPU 메모리 점유율이 낮았다.**
하지만 rank가 낮을 수록 중간의 parameter들이적어 데이터 표현에 제약이 있어서 그런지 **rank가 낮을 수록 loss는 높아졌다.**
학습하려는 데이터가 너무 복잡해서 높은 parameter가 필요한게 아니라면 시간과 GPU가 허용되는 범위 안에서 적당한 rank값을 정하는게 중요한거 같다.
