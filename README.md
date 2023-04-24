# Template Repo for ML Project

This template repo will give you a good starting point for your second project. Besides the files used for creating a virtual environment, you will find a simple example of how to build a simple model in a python script. This is maybe the simplest way to do it. We train a simple model in the jupyter notebook, where we select only some features and do minimal cleaning. The output is then stored in simple python scripts.

The data used for this is: [coffee quality dataset](https://github.com/jldbc/coffee-quality-database).

---
## Requirements and Environment

Requirements:
- pyenv with Python: 3.9.8

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


# Variable definitions

Client:

Client_id: Unique id for client
District: District where the client is
Client_catg: Category client belongs to
Region: Area where the client is
Creation_date: Date client joined
Target: fraud:1 , not fraud: 0
Invoice data

Client_id: Unique id for the client
Invoice_date: Date of the invoice
Tarif_type: Type of tax
Counter_number:
Counter_statue: takes up to 5 values such as working fine, not working, on hold statue, ect
Counter_code:
Reading_remarque: notes that the STEG agent takes during his visit to the client (e.g: If the counter shows something wrong, the agent gives a bad score)
Counter_coefficient: An additional coefficient to be added when standard consumption is exceeded
Consommation_level_1: Consumption_level_1
Consommation_level_2: Consumption_level_2
Consommation_level_3: Consumption_level_3
Consommation_level_4: Consumption_level_4
Old_index: Old index
New_index: New index
Months_number: Month number
Counter_type: Type of counter