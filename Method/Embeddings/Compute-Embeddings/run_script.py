from util.experiment import Experiment
from util.data import Data
import traceback
import argparse
import os, sys

base_path = os.path.dirname(os.path.realpath(__file__)).split('Embeddings')[0]

def start(args):
    datasets = [Data(data_dir=data_dir, train_plus_valid=args.train_plus_valid) for data_dir in args.dataset]
    count = 0
    for d in datasets:
        folder_name = args.dataset[count].split('/')[7]
        experiment = Experiment(dataset=d,
                                model=args.model_name,
                                parameters=vars(args), ith_logger='_' + folder_name,
                                store_emb_dataframe=args.store_emb_dataframe, storage_path=base_path+"Embeddings/"+\
                               folder_name)
        print('Storage path: ', base_path+"Embeddings/"+folder_name)
        count += 1
        try:
            experiment.train_and_eval()
            print()
        except RuntimeError as re:
            print(re)
            traceback.print_exc()
            print('Exit.')
            exit(1)
            
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Ivalid boolean value.')


if __name__ == '__main__':
    folders = [f for f in os.listdir(base_path+"Datasets/") if not f.startswith('.')]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ConEx')
    parser.add_argument('--num_of_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--scoring_technique', default='KvsAll',
                        help="KvsAll technique or Negative Sampling. For Negative Sampling, use any positive integer as input parameter")
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--optim', type=str, default='RMSprop', help='Choose optimizer: Adam or RMSprop')
    parser.add_argument('--decay_rate', default=None)
    parser.add_argument('--train_plus_valid', default=False)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=12.0, help='Distance parameter')
    parser.add_argument('--hidden_dropout', type=float, default=0.1)
    parser.add_argument('--feature_map_dropout', type=float, default=0.1)
    parser.add_argument('--num_of_output_channels', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument("--dataset", nargs='+', type=str, default=folders)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
    parser.add_argument('--store_emb_dataframe', type=str2bool, const=True, default=True, nargs='?', help="Whether to store the embeddings")
    args = parser.parse_args()
    args.dataset = [base_path+'Datasets/'+f+'/Triples/' for f in args.dataset]
    start(args)
