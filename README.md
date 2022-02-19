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
#### S-POP
* exe: `python src/train_s_pop.py`   
* List of Arguments accepted:  
--data_folder String of the directory to the folder containing the dataset.  
--train_data Name of the training dataset file (Default = train.txt)  
--valid_data Name of the validation dataset file (Default = valid.txt)   
--k_eval Value of K used during Recall@K and MRR@K Evaluation (Default = 20)  

#### GRU4REC
* exe: `python src/train.py` 
#### Bi-GRU4REC
* exe: `python src/train.py  --is_biderectional`  
 
List of Arguments accepted (for both GRU4REC and Bi-GRU4REC):  
--hidden_size Number of Neurons per Layer (Default = 100)  
--num_layers Number of Hidden Layers (Default = 1)  
--batch_size Batch Size (Default = 50)  
--dropout_input Dropout ratio at input (Default = 0)  
--dropout_hidden Dropout at each hidden layer except the last one (Default = 0.5)  
--n_epochs Number of epochs (Default = 10)  
--k_eval Value of K used during Recall@K and MRR@K Evaluation (Default = 20)  
--optimizer_type Optimizer (Default = Adagrad)  
--final_act Activation Function (Default = Tanh)  
--lr Learning rate (Default = 0.01)  
--weight_decay Weight decay (Default = 0)  
--momentum Momentum Value (Default = 0)  
--eps Epsilon Value of Optimizer (Default = 1e-6)  
--loss_type Type of loss function TOP1 / BPR / TOP1-max / BPR-max / Cross-Entropy (Default: TOP1-max)  
--time_sort In case items are not sorted by time stamp (Default = 0)  
--model_name String of model name.  
--save_dir String of folder to save the checkpoints and logs inside it (Default = /checkpoint).  
--data_folder String of the directory to the folder containing the dataset.    
--train_data Name of the training dataset file (Default = train.txt)    
--valid_data Name of the validation dataset file (Default = valid.txt)   
--is_eval Should be used in case of evaluation only using a checkpoint model.  
--load_model String containing the checkpoint model to be used in evaluation.  
--checkpoint_dir String containing directory of the checkpoints folder.  


## Evaluating model
#### S-POP
* exe: `python src/train_s_pop.py`   
* List of Arguments accepted:  
--data_folder String of the directory to the folder containing the dataset.  
--train_data Name of the training dataset file (Default = train.txt)  
--valid_data Name of the validation dataset file (Default = valid.txt)   
--k_eval Value of K used during Recall@K and MRR@K Evaluation (Default = 20)  

#### GRU4REC + Bi-GRU4REC
* exe: `python src/eval.py`
* List of Arguments accepted:  
--data_folder String of the directory to the folder containing the dataset.  
--train_data Name of the training dataset file (Default = train.txt)  
--valid_data Name of the validation dataset file (Default = valid.txt)   
--k_eval Value of K used during Recall@K and MRR@K Evaluation (Default = 20)  
--load_model The path to the pytorch model we want to evaluate (Default = None)

## Results:
All the results can be found [here](https://docs.google.com/spreadsheets/d/1wlwuKIeaMwBFY6iebWhtnU_rDA5xmZxO8wUdFyTjI9g/edit?usp=sharing)
