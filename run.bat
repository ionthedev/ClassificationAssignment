@echo off
mkdir models
mkdir data
mkdir results
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py
