# ASR on THCHS30
## Hybrid DNN-HMM Model
* Acoustic model: 利用HMM聲學建模，隱馬爾可夫模型輸出從各個幀計算而得的聲學特徵，將wav檔轉成mfcc
* Language model:  Seq2seq model (RNN/GRU/LSTM)

### Requirements: 
* **Tensorflow r1.5.1**
* Python 3.6
* Numpy 1.13.3
* [librosa](https://github.com/librosa/librosa)

### Usage
1. Download dataset from:
```
http://www.openslr.org/18/
```

2. extract the .tgz file, and place them to the path below: 
```
./data_thchs30/data/
```

3. Run training:
```
python3 ASR_THCHS30.py --train
```
