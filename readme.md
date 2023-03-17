to train and do 4-Fold Cross validation, just run \\
```python train.py --data 30 --in_channel 3 --eval --cv 4 --epochs 5```

To create a dataset for NN with a windowsize of 360, DWT level of 2 and jumps of 1, run 


*Method DWT* ``` python multithread.py --w 360 --jump 10  --level 2 --method dwt ```
*Method SSA* ``` python multithread.py --w 360 --jump 10 --w_ssa 50 --thresh 0.9 --method ssa ```
