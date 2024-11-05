
### Rank 크기 별 메모리 점유
#### print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')로 찍은 기준
- Rank 8: 3.9 GB
- Rank 128: 4.6 GB
- Rank 256: 
#### wandb 기준
