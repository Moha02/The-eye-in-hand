# The-eye-in-hand

The Eye in Hand: Bayesian Action Prediction Experiment
Project Overview

This repository contains the implementation of the first experiment detailed in the paper "The Eye in Hand". The experiment aims to predict the intended target of a hand movement using Bayesian methods, considering various cues such as gaze congruency, hand preshapes, and arm trajectory vectors.
Repository Structure


.
├── README.md                 # Project documentation
├── ambrosini15actionPrediction.pdf  # Reference paper
├── experiment_data.csv       # CSV file with experiment data
├── main.py                   # Script to run the experiment
└── model.py                  # Bayesian Action Prediction model implementation

Implementation Details

The project is primarily implemented in Python, leveraging the numpy library for numerical operations and matplotlib for visualizations.
Bayesian Action Prediction Model

The model is implemented in model.py.

Script to Run the Experiment

The main.py script initializes the Bayesian Action Prediction model with 1000 trials and runs the experiment. It also generates a report summarizing the experiment's results.
Usage

To run the experiment, follow these steps:

Clone the repository:

git clone <repository_url>
cd <repository_directory>

Install dependencies:

pip install numpy matplotlib

Run the experiment:

python main.py


Reference

For more details on the theoretical background and methodology, please refer to the paper:

    Ambrosini, E., Costantini, M., & Sinigaglia, C. (2015). The eye in hand: Predicting others’ behavior by integrating multiple action cues. Neuropsychologia, 70, 130-136.


For any questions or issues, please contact me.
