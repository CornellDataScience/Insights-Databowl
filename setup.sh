#!/usr/bin/env bash

# this script installs all python dependencies
# and executes the data processing pipeline to create
# all the csv data files required in this project

# install python dependencies
pip3 install pandas matplotlib seaborn tqdm

# unzip raw data
unzip data/nfl-big-data-bowl-2020.zip -d data
cp data/train.csv data/raw_data.csv

# data cleaning
python3 data-exploration/data-cleaning.py

# generate vis play
cp data/clean_data.csv data/test_data.csv
cd vis
python3 generate-json.py
