import transformers
import os, sys
from datasets import Dataset
import argparse
import shutil
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from helper_functions import get_data
sys.path.append(os.path.dirname(os.path.realpath(__file__)).split('transformers')[0])
from ontolearn import KnowledgeBase


parser = argparse.ArgumentParser()
parser.add_argument("--kb", type=str, default="carcinogenesis", help="Name of the knowledge base")
parser.add_argument("--model_path_or_name", type=str, default="t5-small", help="Name or path to the transformer model")
parser.add_argument("--model_name", type=str, default="t5-small", help="Name of the transformer model")
parser.add_argument("--path_tokenizer", type=str, default="", help="Path to the pretrained tokenizer")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=3.5e-5, help="Learning rate")
parser.add_argument('--model_max_length', type=int, default=1024, help='The maximum sequence length')
#parser.add_argument("--use_pretrained_tokenizer", type=bool, default=True, help="Whether to use a pretrained tokenizer")
args = parser.parse_args()

data = get_data(f"../Datasets/{args.kb}/Train_data")
kb_path = f'../Datasets/{args.kb}/{args.kb}.owl'
data = data.train_test_split(test_size=0.2, seed=42)

kb = KnowledgeBase(path=kb_path)
Vocab = list(map(lambda x: x.get_iri().as_str().split("/")[-1].split('#')[-1], kb.individuals())) + \
['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', ')', '(', '.', 'StartPositive', 'EndPositive', 'StartNegative', 'EndNegative']

if not os.path.exists(args.model_path_or_name):
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train_from_iterator(Vocab, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.pad_token = "[PAD]"
    
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, model_max_length=args.model_max_length)
    tokenizer.add_tokens(Vocab)
#




#tokenizer.add_tokens(['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥'])

source_lang = "lang1"
target_lang = "lang2"
prefix = ""
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path_or_name)
model.resize_token_embeddings(len(tokenizer))


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

output_dir = "./results_"+args.kb + "_" + args.model_name.replace("-", "_").split("/")[-1]
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=args.epochs,
    fp16=True,
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
shutil.rmtree(output_dir)
trainer.save_model(f"{output_dir}/model")