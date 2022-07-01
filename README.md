# Neural Class Expression Synthesis (NCES)
Implementation of neural class expression synthesizers (NCES)

![ncel-dlo](ncel-dlo.png)

## Installation

Clone this repository:
```
git clone https://github.com/dice-group/NCES2.git
``` 

Make sure Anaconda3 is installed in your working environment then run the following to install all required librairies for NCES:
```
conda env create -f environment.yml
```
A conda environment (nces) will be created. Next activate the environment:
``` conda activate nces```

Download and extract Datasets by running the following ``` bash download_data ```

To run search based algorithms CELOE, OCEL, ELTL and ECII, install Java 8+ and Maven 3.6.3+

Dowload DL-Learner-1.4.0 from [github](https://github.com/SmartDataAnalytics/DL-Learner/releases) and extract it into the directory Method

Clone DL-Foil from [bitbucket](https://bitbucket.org/grizzo001/dl-foil.git) into Method

## Reproducing the reported results

### NCES (Ours)


*Open a terminal and navigate into Method/transformer/* ``` cd NCES2/Method/transformers/```
- Reproduce training NCES: ``` python train.py ``` with the following options

``` 
--kb KB               Name of the knowledge base
  --model_path_or_name MODEL_PATH_OR_NAME
                        Name or path to the transformer model
  --model_name MODEL_NAME
                        Name of the transformer model
  --path_tokenizer PATH_TOKENIZER
                        Path to the pretrained tokenizer
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --model_max_length MODEL_MAX_LENGTH
                        The maximum sequence length

```

- To reproduce evaluation results on concept learning, please open the jupyter notebook file ReproduceNCES.ipynb

*Remark: --kb is one of carcinogenesis, mutagenesis, family-benchmark, semantic_bible, vicodi*

Costum knowledge bases can also be added. To do this, create a directory (knowledge base name) in Datasets, and add its owl format (make sure the owl file has the same name as your folder name). Create data generation file by following examples in
generators/train_data/. Run the file to generate training and test data. Train the transformer model by running ``` python train.py --kb your_kb_name ```.

### DL-Learner (Lehmann et al.)

*Open a terminal and navigate into Method/dllearner/* ``` cd NCES2/Method/dllearner/```
- Reproduce CELOE, OCEL, and ELTL concept learning results: ``` python reproduce_dllearner_experiment.py --algo --kb --max_runtime --num_probs```

### DL-Foil (Fanizzi et al.)

*Open a terminal and navigate into Method/dl-foil/* ``` cd NCES2/Method/dl-foil/```

- Run mvn package

- Copy `generate_dlfoil_config_all_kbs.py` into dl-foil and run `python generate_dlfoil_config_all_kbs.py` to prepare configuration files for all knowledge bases

- Reproduce concept learning results: ` mvn -e exec:java -Dexec.mainClass=it.uniba.di.lacam.ml.DLFoilTest -Dexec.args=DL-Foil2/kb_config.xml `

### ECII (Sarker et al.)

*Open a terminal and navigate into Method/ecii/* ``` cd NCES2/Method/ecii/```

- Run `python generate_config_ecii.py --kb "knowledge base name(s)" ` to prepare configuration files

- To start concept learning, run `java -Xms2g -Xmx8g -Xss1g -jar ecii_v1.0.0.jar -b kb/`

- Run `python parse_ecii_output.py --kb "knowledge base name(s)" ` to parse the output and save the results such as f_measure and runtime

## Acknowledgement 
We based our implementation on the open source implementation of [ontolearn](https://docs--ontolearn-docs-dice-group.netlify.app/). We would like to thank the Ontolearn team for the readable codebase.
