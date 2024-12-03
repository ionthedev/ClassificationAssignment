#!/bin/bash
mkdir models
mkdir data
mkdir results
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py

