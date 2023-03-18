To train and do 4-Fold Cross validation, just run \\
```python train.py --data training_data/w30level2jump10 --in_channel 3 --eval --epochs 3 --cv 4 --resizing 50 --kernel 3```

To Cross validation for all dataset and store the results
```list_of_data = os.listdir("training_data")```
```for data in  list_of_data:```
    ```!(python train.py --data training_data/{data} --in_channel 3 --eval --epochs 3 --cv 4  --resizing 50 --kernel 3) > results/{data}.out ```

One you have the best dataset. To split it in train and test set run 
```python train_test_split.py --data training_data/w360level3jump10 --ratio 0.8 --output evaluation train.py --data training_data/w30level2jump10 --in_channel 3  --epochs 3  --resizing 50```

To train on training set and save the model
```python train.py --data training_data/w30level2jump10 --in_channel 3  --epochs 3  --resizing 50 --kernel 3 --save_path model```

To evaluate the model on test set

```python evaluate.py --data evaluation/test --model model/model__epoch_2.pth --in_channel 3 --resizing 50 --out_dir test_results ```

check some outputs in test_results folder


To create a dataset for NN with a windowsize of 360, jumps of 10, run 


*Method DWT + Fourier spectrogram* ``` python multithread.py --w 360 --jump 10  --level 2 --method dwt --fourier True ```

*Method DWT + Wavelet scalogram* ``` python multithread.py --w 360 --jump 10  --level 2 --method dwt --fourier False ```

*Method SSA + Fourier spectrogram* ``` python multithread.py --w 360 --jump 10 --w_ssa 50 --thresh 0.9 --method ssa --fourier True ```

*Method SSA + wavelet scalogram* ``` python multithread.py --w 360 --jump 10 --w_ssa 50 --thresh 0.9 --method ssa --fourier False ```
