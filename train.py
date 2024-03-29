import matplotlib.pyplot as plt
import torch, pandas as pd, numpy as np
import os, json
import argparse
import random

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


from utils.experiment import Experiment
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--ablation_type', type=str, default='', choices=['alc_1', 'alchiqd_0'], help='Which ablation to perform: alc with improved data generator (alc_1) or alchiqd without improved data generator (alchiqd_0)')
parser.add_argument('--all_num_inds', type=int, nargs='+', default=[32, 64, 128], help='Number of induced instances provided as a list')
parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
parser.add_argument('--decay_rate', type=float, default=0.0, help='Decay rate for the optimizer')
parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout rate in neural networks')
parser.add_argument('--embedding_dim', type=int, default=50, help='Number of embedding dimensions')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--feature_map_dropout', type=float, default=0.1, help='Feature map dropout probability')
parser.add_argument('--final', type=str2bool, default=False, help='Whether to train the concept synthesizer on test+train data')
parser.add_argument('--grad_clip_value', type=float, default=5.0, help='Gradient clip value')
parser.add_argument('--hidden_dropout', type=float, default=0.1, help='Hidden dropout probability during embedding computation')
parser.add_argument('--input_dropout', type=float, default=0.0, help='Input dropout probability for embedding computation')
parser.add_argument('--kb_emb_model', type=str, default='ConEx', help='Embedding model name')
parser.add_argument('--kbs', type=str, nargs='+', default=['carcinogenesis'], help='Knowledge base name. Check the folder datasets to see all available knowledge bases')
parser.add_argument('--kernel_size', type=int, default=4, help='Kernel size in ConEx')
parser.add_argument('--knowledge_base_path', type=str, default="", help='Path to KB owl file')
parser.add_argument('--learner_name', type=str, default="SetTransformer", choices=['SetTransformer'], help='Neural model')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--ln', type=str2bool, default=False, help='Whether to use layer normalization')
parser.add_argument('--load_pretrained', type=str2bool, default=False, help='Whether to load pretrained models')
parser.add_argument('--max_length', type=int, default=48, help='Maximum length of class expressions')
parser.add_argument('--models', type=str, nargs='+', default=['SetTransformer'], help='Neural models')
parser.add_argument('--num_examples', type=int, default=1000, help='Total number of examples for concept learning')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_inds', type=int, default=32, help='Number of induced instances')
parser.add_argument('--num_of_output_channels', type=int, default=8, help='Number of output channels in ConEx')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of seed components in the output')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers to use to load training data')
parser.add_argument('--opt', type=str, default='Adam', help='Name of the optimizer to use')
parser.add_argument('--path_to_triples', type=str, default="", help='Path to KG (result of the conversion of KB to KG)')
parser.add_argument('--proj_dim', type=int, default=128, help='The projection dimension for examples')
parser.add_argument('--save_model', type=str2bool, default=True, help='Whether to save the model after training')
parser.add_argument('--test', type=str2bool, default=False, help='Whether to evaluate the concept synthesizer on the test data during training')
args = parser.parse_args()
    
print("Setting: ", vars(args))

with open(f"settings.json", "w") as setting:
    json.dump(vars(args), setting)

for kb in args.kbs:
    data_path = f"datasets/{kb}"
    data_train_path = f"{data_path}/Train_data/Data_{args.ablation_type}.json" if args.ablation_type else f"{data_path}/Train_data/Data.json"
    with open(data_train_path, "r") as file:
        data_train = json.load(file)

    if not args.ablation_type:
        data_test_path = f"{data_path}/Test_data/Data.json"
        with open(data_test_path, "r") as file:
            data_test = json.load(file)
    else:
        data_test = []
    args.knowledge_base_path = f"{data_path}/{kb}.owl"
    args.path_to_triples = f"{data_path}/Triples/"
    for num_inds in args.all_num_inds:
        args.num_inds = num_inds
        experiment = Experiment(data_train, data_test, args)
        final = args.final
        test = args.test
        if args.final and not args.ablation_type:
            data_train = data_train + data_test
            test = False
        experiment.train_all_nets(args.models, data_train, data_test, epochs=args.epochs, test=test, save_model=args.save_model, save_path=data_path, ablation_type=args.ablation_type,
                                  kb_emb_model=args.kb_emb_model, optimizer=args.opt, record_runtime=True, final=final)
