#!/bin/sh
python ./train_data/generate_training_data_carcinogenesis_kg.py
python ./train_data/generate_training_data_family_benchmark_kg.py
python ./train_data/generate_training_data_mutagenesis_kg.py
python ./train_data/generate_training_data_semantic_bible_kg.py
python ./train_data/generate_training_data_vicodi_kg.py