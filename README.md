# Deep Generalised Canonical Correlation Analysis (DGCCA)

This repository provides a modular PyTorch implementation of DGCCA, designed to learn shared representations from multi-view data.
The original DGCCA article is the following: 
Adrian Benton, Huda Khayrallah, Biman Gujral, Dee Ann Reisinger, Sheng Zhang, and Raman Arora. Deep Generalized Canonical Correlation Analysis. The 4th Workshop on Representation Learning for NLP. 2019

It is available at: https://www.aclweb.org/anthology/W19-4301/

A Theano implementation of DGCCA is available at: https://bitbucket.org/adrianbenton/dgcca-py3

## 📂 Project Structure

This repository contains the following core components:

* **`src/`**: A directory containing the modular Python scripts:
    * `model.py`: Defines the `DeepGCCA` architecture and the `generalised_gcca_loss` loss function.
    * `utils.py`: Helper functions for data loading and reproducibility (seeding).
    * `main.py`: The execution script that orchestrates the training loop and saves artifacts.
* **`run_dgcca.sh`**: A Bash script to run the entire pipeline with customizable hyperparameters (Learning Rate, Epochs, Latent Dimensions) without editing Python code.
* **`generate_data.py`**: A utility script to create synthetic multi-view datasets for testing and validation.
* **`requirements.txt`**: A list of the Python libraries required to run the project (PyTorch, NumPy, Pandas, etc.).
* **`dgcca_colab.ipynb`**: The original Google Colab notebook used for initial prototyping and experimentation.

## 🛠 Setup and Installation

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python3.12 -m venv env

# Activate the environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

To generate synthetic data and train the model using the modular pipeline, execute the following commands in your terminal:

```bash
# 1. Generate the synthetic multi-view data
python3 generate_data.py

# 2. Run the DGCCA training pipeline
chmod +x run_dgcca.sh
./run_dgcca.sh
```
Note on using your own data: To use this implementation with your own datasets, ensure your views are organised into separate .csv files. All files must have the same number of samples (rows) in the same order, as the model learns to correlate observations across views.

## 📊 Expected Outputs

Upon successful execution, the pipeline creates a `/results` directory locally. These files are ignored by Git to keep the repository clean, but will be available on your machine:

* **`shared_G.csv`**: The learned shared latent representation (embeddings).
* **`encoded_view_1.csv`, `encoded_view_2.csv`, etc.**: The projected embeddings for each individual view, mapped into the common latent space.
* **`loss_history.csv`**: A CSV file containing the training loss for each epoch.
* **`model_weights.pth`**: The saved state dictionary of the trained neural network.
* **`config.json`**: A record of the specific hyperparameters used for the run.
