to train and do 4-Fold Cross validation, just run \\
```python train.py --data 30 --in_channel 3 --eval --cv 4 --epochs 5```

To create a dataset for NN with a windowsize of 360, DWT level of 2 and jumps of 1, run ``` python multithread.py --w 360 --level 2 --jump 1 ```
