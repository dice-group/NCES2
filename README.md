# NCES2
Neural Class Expression Synthesis in ALCHIQ(D)


## Installation

Make sure Anaconda3 is installed in your working environment then run the following to install all required librairies for NCES2:
```
conda env create -f environment.yml
```

A conda environment (nces2) will be created. Next activate the environment:

```
conda activate nces2 
```

Also install Ontolearn: 

``` 
git clone https://github.com/dice-group/Ontolearn.git
```
then

``` 
cd Ontolearn, python -c 'from setuptools import setup; setup()' develop
```

- To run search based algorithms CELOE, ELTL and ECII, first install Java 8+ and Maven 3.6.3+

## Reproducing the reported results

### NCES2 (Ours)


*Open a terminal in NCES2/*


- To reproduce NCES2 results in Table 4: ``` python reproduce_nces2.py ```. Use -h for more options, e.g., use `--kb_emb_model Distmult` for the results in the Appendix

- To reproduce NCES2 results in Table 6 ``` python reproduce_nces2_robustness.py ```. Use -h for more options.


### DL-Learner (Lehmann et al.)

*Open a terminal and navigate into dllearner/* ``` cd NCES2/dllearner ```

- Reproduce CELOE and ELTL concept learning results: ``` python reproduce_dllearner_experiment.py --algo --kbs --max_runtime ```

*Remark: --kbs is one of carcinogenesis, mutagenesis, semantic_bible, or vicodi*


### ECII (Sarker et al.)

*Open a terminal and navigate into ecii/* ``` cd NCES2/ecii/```
- Download the jar file `ecii_v1.0.0.jar` into ecii/

- Run `python generate_config_ecii.py ` to prepare configuration files

- To start concept learning, run `java -Xms2g -Xmx8g -Xss1g -jar ecii_v1.0.0.jar -b kb/`

- Run `python parse_ecii_output.py ` to parse the output and save the results such as f_measure and runtime

* Here kb is one of carcinogenesis, mutagenesis, semantic_bible, or vicodi *

### EvoLearner (Heindorf et al.)

*Open a terminal and navigate into evolearner/* ``` cd NCES2/evolearner/ ```

- Run `python run_evolearner.py `. Use options to select the knowledge base, save results, or enable logging. Example `python run_evolearner.py --kbs carcinogenesis --save_results True` 


## Training NCES2 on our datasets:

- Reproduce training NCES2: ``` python train.py ``` optionally with the following options:

``` 
  --kbs {carcinogenesis,mutagenesis,semantic_bible,vicodi} [{carcinogenesis,mutagenesis,semantic_bible,vicodi} ...]
                        Knowledge base name
  --models MODELS [MODELS ...]
                        Neural models
  --kb_emb_model KB_EMB_MODEL
                        Embedding model name
  --load_pretrained LOAD_PRETRAINED
                        Whether to load the pretrained model on carcinogenesis
  --learner_name {LSTM,GRU,SetTransformer}
                        Neural model
  --knowledge_base_path KNOWLEDGE_BASE_PATH
                        Path to KB owl file
  --path_to_csv_embeddings PATH_TO_CSV_EMBEDDINGS
                        KB embedding path
  --learning_rate LEARNING_RATE
                        Learning rate
  --embedding_dim EMBEDDING_DIM
                        Number of embedding dimensions
  --input_size INPUT_SIZE
                        Number of embedding dimensions in the input
  --num_workers NUM_WORKERS
                        Number of workers to use to load training data
  --proj_dim PROJ_DIM   The projection dimension for examples
  --num_inds NUM_INDS   Number of induced instances
  --num_heads NUM_HEADS
                        Number of attention heads
  --num_seeds NUM_SEEDS
                        Number of seed components in the output
  --num_examples NUM_EXAMPLES
                        Total number of examples for concept learning
  --ln LN               Whether to use layer normalization
  --decay_rate DECAY_RATE
                        Decay rate for the optimizer
  --grad_clip_value GRAD_CLIP_VALUE
                        Gradient clip value
  --opt OPT             Name of the optimizer to use
  --rnn_n_layers RNN_N_LAYERS
                        Number of recurrent network layers
  --max_length MAX_LENGTH
                        Maximum length of class expressions
  --drop_prob DROP_PROB
                        Dropout rate in neural networks
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Training batch size
  --cross_validate CROSS_VALIDATE
                        Whether to use a 10-fold cross-validation setting
  --shuffle_examples SHUFFLE_EXAMPLES
                        Whether to shuffle positive and negative examples in the dataloader
  --test TEST           Whether to evaluate the concept synthesizer on the test data during training
  --final FINAL         Whether to train the concept synthesizer on test+train data
  --save_model SAVE_MODEL
                        Whether to save the model after training
  ```


## Bring your own data

To train NCES2 on a new knowledge base, create a new folder under datasets and add the OWL format of the knowledge base in the folder. Make sure the owl file has the same name as the folder you created. Follow the 3 steps below to train NCES2 on your knowledge base.

- (1) Generating training data for NCES2: `cd generators/` then ` python generate_data.py --kbs your_folder_name `. Use -h for more options. For example, use `--num_rand_samples 500` combined with `--refinement_expressivity 0.6` to increase the amount of training data.

- (2) Convert knowledge base to knowledge graph: ```cd generators ``` then ``` python kb_to_kg.py --kbs your_folder_name ```

- (3) Training NCES2 on your data: `cd NCES2/ ` then ` python train.py --kbs your_folder_name `. Use -h to see more options for training, e.g., `--batch_size` or `--epochs`

