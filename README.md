# NCES2
Neural Class Expression Synthesis in ALCHIQ(D)


## Installation

Clone this repository: 

```
git clone --branch 0.5.4 --depth 1 https://github.com/dice-group/Ontolearn.git
```

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

- First download datasets and pretrained models: `cd NCES2/` then ```bash download_data```

### NCES2 (Ours)


*Open a terminal in NCES2/*

- To reproduce NCES2 results (Table 4): ``` python reproduce_nces2.py ```. Use -h for more options, e.g., use `--kb_emb_model Distmult` for the results in the Appendix

- To reproduce ablation results (Table 6): ``` python reproduce_nces2_ablation.py ```. Use `--ablation_type alc_1` or `--ablation_type alchiqd_0` to select the ablation to run. Use -h to view available options.



### NCES1 (Kouagou et al.)

*Open a terminal in NCES2/*

- To run NCES1 (Table 6): ``` python reproduce_nces1.py ```


### DL-Learner (Lehmann et al.)

*Open a terminal and navigate into dllearner/* ``` cd NCES2/dllearner ```

- Reproduce CELOE and ELTL concept learning results: ``` python reproduce_dllearner_experiment.py --algo --kbs --max_runtime ```

*Remark: --kbs is one of carcinogenesis, mutagenesis, semantic_bible, or vicodi*


### ECII (Sarker et al.)

*Open a terminal and navigate into ecii/* ``` cd NCES2/ecii/```

- Run `bash init` to prepare subdirectories (copy ontologies into the directory ecii) 

- Download the jar file `ecii_v1.0.0.jar` into ecii/. The file can be found here: https://github.com/md-k-sarker/ecii-expr/blob/master/system/ecii_v1.0.0.jar

- Run `python generate_config_ecii.py ` to prepare configuration files

- To start concept learning, run `java -Xms2g -Xmx8g -Xss1g -jar ecii_v1.0.0.jar -b kb/`

- Run `python parse_ecii_output.py ` to parse the output and save the results such as f_measure and runtime

* Here kb is one of carcinogenesis, mutagenesis, semantic_bible, or vicodi *

### EvoLearner (Heindorf et al.)

*Open a terminal and navigate into evolearner/* ``` cd NCES2/evolearner/ ```

- Run `python run_evolearner.py `. Use options to select the knowledge base, save results, or enable logging. Example `python run_evolearner.py --kbs carcinogenesis --save_results True` 


## Training NCES2 on our datasets:

- Reproduce training NCES2: ``` python train.py ```. Use -h for more options.


## Bring your own data

To train NCES2 on a new knowledge base, create a new folder under datasets and add the OWL format of the knowledge base in the folder. Make sure the owl file has the same name as the folder you created. Follow the 3 steps below to train NCES2 on your knowledge base.

- (1) Generate training data for NCES2: `cd generators/` then ` python generate_data.py --kbs your_folder_name `. Use -h for more options. For example, use `--num_rand_samples 500` combined with `--refinement_expressivity 0.6` to increase the amount of training data.

- (2) Convert knowledge base to knowledge graph: ```cd generators ``` then ``` python kb_to_kg.py --kbs your_folder_name ```

- (3) Training NCES2 on your data: `cd NCES2/ ` then ` python train.py --kbs your_folder_name `. Use -h to see more options for training, e.g., `--batch_size` or `--epochs`

