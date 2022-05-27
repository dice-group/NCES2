import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import sys, os, json, random

base_path = os.path.dirname(os.path.realpath(__file__)).split('reproduce_results')[0]
sys.path.append(base_path)

from helper_classes.experiment import Experiment
from sklearn.model_selection import train_test_split
from util.data import Data
import json

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
data_path = base_path+"Datasets/semantic_bible/Train_data/Data.json"
with open(data_path, "r") as file:
    data = json.load(file)
data = list(data.items())
path_to_triples = base_path+"Datasets/semantic_bible/Triples/"
triples = Data({"path_to_triples":path_to_triples})

max_num_atom_repeat = 10
kwargs = {"learner_name":"GRU", "emb_model_name":"", 'knowledge_graph_path': base_path+"Datasets/semantic_bible/semantic_bible.owl",
          "pretrained_embedding_path":base_path+"Datasets/semantic_bible/Model_weights/ConEx_GRU.pt",
          "pretrained_concept_synthesizer":base_path+"Datasets/semantic_bible/Model_weights/GRU.pt", 
          "path_to_csv_embeddings":base_path+"Embeddings/semantic_bible/ConEx_entity_embeddings.csv",
          "learning_rate":0.0001, "decay_rate":0, 'grad_clip_value':5., "path_to_triples":path_to_triples, 'max_num_atom_repeat': max_num_atom_repeat,
          'index_score_upper_bound':10., 'index_score_lower_bound_rate': 0.8, 'max_num_tokens':30,
          "random_seed":seed, "embedding_dim":20, "num_entities":len(triples.entities),
          "num_relations":len(triples.relations), "num_examples":362, "input_dropout":0.0, 'drop_prob':0.1,
          "kernel_size":4, "num_of_output_channels":8, "feature_map_dropout":0.1,
          "hidden_dropout":0.1, "rnn_n_layers":2, 'input_size':40,
          'rnn_n_hidden':100,'seed':10, 'kernel_w':5, 'kernel_h':11, 'stride_w':1, 'stride_h':3, 'conv_out':1408}

Models = ["GRU_As_MT", "LSTM_As_MT"] #["GRU", "LSTM", "CNN"]

experiment = Experiment(kwargs)

data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

final = False
test = True
cross_validate = False
record_runtime = True
save_model = True
if final:
    data_train = data
    test = False
    cross_validate = False
experiment.train_all_nets(Models, data_train, data_test, epochs=1000, cs_batch_size=128, tc_batch_size=256, kf_n_splits=10, cross_validate=cross_validate, test=test, save_model = save_model, include_embedding_loss=True, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=record_runtime)
