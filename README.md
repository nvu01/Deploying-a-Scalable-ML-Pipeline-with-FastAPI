Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## Repositories
This repository uses GitHub Actions for continuous integration (CI). The workflow runs pytest and flake8 on push and requires both to pass without error.
The workflow is defined in the .github/workflows/manual.yml file.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* ml/data.py: Contains data preprocess function.
* ml/model.py: Contains functions that train a machine learning model (RandomForestClassifier), save the model and compute performance metrics on the test dataset and categorical slices.
* train_model.py: Applies functions from data.py and model.py to train the model on the clean data.
* test_ml: 3 unit tests for codes used to train the model.
* model_card_template.md: a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI which implements:
    * GET on the root giving a welcome message.
    * POST that does model inference.
