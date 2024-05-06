
from bayesian_action_prediction.model import BayesianActionPrediction

# Initialize the Bayesian action prediction experiment, and generate the report.
experiment = BayesianActionPrediction(trials=1000)
experiment.run_experiment()
experiment.generate_report()