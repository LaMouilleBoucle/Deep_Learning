import os

if __name__ == "__main__":
  for t_minus_one in range(4,22):
    os.system('python train.py --input_dim 10 --input_length %i' %(t_minus_one))
    print('Done: t_minus_one')
