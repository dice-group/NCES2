import transformers
from tqdm import tqdm
import os
from datasets import Dataset
import argparse
import shutil

def get_data(data_path):
    import json
    from collections import defaultdict
    if not os.path.isfile(data_path):
        with open(f"{data_path}/Data.json") as file:
            data = json.load(file)
    else:
        with open(f"{data_path}") as file:
            data = json.load(file)
    new_data = defaultdict(lambda: [])
    for i,concept in tqdm(enumerate(data)):
        positives = data[concept]['positive examples']
        negatives = data[concept]['negative examples']
        #if "." in concept:
        #    concept = concept.replace(".", " . ")
        new_data["id"].append(i)
        new_data["translation"].append({
                         "lang1": "Positive " + " ".join(positives)+" Positive "
                         + "Negative " + " ".join(negatives)+" Negative",
                        "lang2": concept})
    return Dataset.from_dict(new_data)


parser = argparse.ArgumentParser()
parser.add_argument("--kb", type=str, default="carcinogenesis", help="Name of the knowledge base")
parser.add_argument("--model", type=str, default="t5-small", help="Name of the transformer model")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=3.5e-5, help="Learning rate")
args = parser.parse_args()

data = get_data(f"../Datasets/{args.kb}/Train_data")

data = data.train_test_split(test_size=0.2)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model)

source_lang = "lang1"
target_lang = "lang2"
prefix = "translate instance data to class expression"


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

output_dir = "./results_"+args.kb + "_" + args.model.replace("-", "_").split("/")[-1]
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