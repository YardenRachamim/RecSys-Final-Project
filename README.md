# RecSys-Final-Project

* This is the code of ResSys course final project in RU 2022.  
* In this repo you can find an implementation of the GRU4REC model([SESSION-BASED RECOMMENDATIONS WITH
RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1511.06939.pdf)) and extension of GRU4REC to a biderectional model (our improvment). 
* This code is heavily influenced by https://github.com/hungthanhpham94/GRU4REC-pytorch#readme (Pham Thanh Hung and Mohamed Maher)


## Dataset
1. **YOUCHOOSE** dataset can be found [here](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
2. **DIGINETICA** dataset can be found [here](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)

* The data is being preprocessed according to the article.  
The preprocessed data can be found in the links attached: [YOUCHOOSE](data/youchoose), [DIGINETICA](data/digenetica).  
* If one wants to reproduce the preprocess running, he needs to run [this](src/data_preprocess) with the raw data path arguments.  
Where for the digenetica the relevant data file is: `dataBefore="path to train-item-views.csv"` and for youchoose: `dataBefore = "path to yoochoose-clicks.dat"`
Example: `python preprocessing_yoochoose.py`


## Train model
There are 3 models in the project:
1. S-POP
2. GRU4REC
3. Bi-GRU4REC
The 3 of the models can be trained with each of the datasets. 
Examples:
1. S-POP exe: `python src/train_s_pop.py --data_path data/<dataset name>` 
