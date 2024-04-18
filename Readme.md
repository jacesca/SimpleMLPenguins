# A simple machine learning project using scikit learn
The objective of this model is to predict the specie of the penguin.

Features:
- Preprocessing data using scikit-learns
- Pipelines to streamline the process
- Modeling

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone 
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Run ML model
- python model.py


## Others
- Proyect in GitHub: https://github.com/jacesca/SimpleMLPenguins
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > flask_env.yml
```
- For coding style
```
black model.py
flake8 model.py
```