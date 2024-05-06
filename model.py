import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BayesianActionPrediction:
    def __init__(self, trials=1000, object_positions=None):
        self.trials = trials
        self.gaze_congruency = np.random.randint(2, size=trials)
        self.hand_preshapes = np.random.randint(2, size=trials)
        self.actual_target = np.random.randint(2, size=trials)

        
        # Define positions for two objects if not provided
        if object_positions is None:
            self.object_positions = [(4, 3), (-2, 2)]  # Example positions for two objects
        else:
            self.object_positions = object_positions
        
        # Initialize array to store vectors pointing towards objects
        self.arm_vectors = np.zeros((trials, 2))  # Each vector is an (x, y) pair
        
        # Randomly choose an object for each trial and calculate the vector
        for i in range(trials):
            target_object_pos = self.object_positions[self.actual_target[i]]
            # Generate a random starting position for the arm for each trial
            arm_start_pos = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            self.arm_vectors[i] = np.array(target_object_pos) - np.array(arm_start_pos)

        
        
        self.accuracy_history = []
        self.correct_predictions = []

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')  # Initialize an empty plot line
        self.ax.set_xlim(0, trials)  # X-axis from 0 to total number of trials
        self.ax.set_ylim(0, 100)

    def init(self):
        """Initialize the background of the plot."""
        self.line.set_data([], [])
        return self.line,

    def predict(self, gaze, preshape, arm_vector):
        prior = 0.5
        likelihood_gaze = 0.6 if gaze else 0.4
        likelihood_preshape = 0.7 if preshape else 0.3
        # Use the magnitude of the arm vector as a proxy for trajectory reliability
        likelihood_trajectory = np.linalg.norm(arm_vector) / np.linalg.norm(np.array(self.object_positions).max(axis=0))
        
        combined_likelihood = likelihood_gaze * likelihood_preshape * likelihood_trajectory
        posterior = combined_likelihood * prior / (combined_likelihood * prior + (1 - combined_likelihood) * (1 - prior))
        return 1 if posterior > 0.5 else 0
    
    def calculate_accuracy_for_gaze(self):
        # Assuming gaze confidence > 0.5 implies a prediction towards the target being 1 (true)
        correct = [1 if (gaze > 0.5) == (target == 1) else 0 for gaze, target in zip(self.gaze_congruency, self.actual_target)]
        gaze_accuracy = sum(correct) / len(correct) * 100
        return gaze_accuracy
    def calculate_accuracy_for_preshape(self):
        # Assuming preshape confidence > 0.5 implies a prediction towards the target being 1 (true)
        correct = [1 if (preshape > 0.5) == (target == 1) else 0 for preshape, target in zip(self.hand_preshapes, self.actual_target)]
        preshape_accuracy = sum(correct) / len(correct) * 100
        return preshape_accuracy
    def calculate_accuracy_for_trajectory(self):
        # Placeholder: Assume 50% accuracy as a base, modify based on your model's specifics
        # In practice, this method should evaluate how well the arm vector (direction and magnitude)
        # correlates with correct target predictions.
        trajectory_accuracy = 50.0  # This is a simplistic placeholder
        return trajectory_accuracy
    
    def generate_report(self):
        # Calculate Total Accuracy
        total_accuracy = sum(self.correct_predictions) / len(self.correct_predictions) * 100

        # Accuracy by Cue
        gaze_accuracy = self.calculate_accuracy_for_gaze()
        preshape_accuracy = self.calculate_accuracy_for_preshape()
        trajectory_accuracy = self.calculate_accuracy_for_trajectory()  # Assuming this method is implemented

        # Average Confidence Metrics
        average_gaze_confidence = np.mean(self.gaze_congruency)
        average_preshape_confidence = np.mean(self.hand_preshapes)
        average_trajectory_confidence = np.mean([np.linalg.norm(vec) for vec in self.arm_vectors])

        # Prepare the report
        report = f"""
        Total Accuracy: {total_accuracy:.2f}%
        Accuracy by Cue:
            Gaze: {gaze_accuracy:.2f}%
            Preshape: {preshape_accuracy:.2f}%
            Trajectory: {trajectory_accuracy:.2f}%
        Average Confidence Levels:
            Gaze: {average_gaze_confidence:.2f}
            Preshape: {average_preshape_confidence:.2f}
            Trajectory (norm): {average_trajectory_confidence:.2f}
        """

        # Print or return the report
        print(report)
        return report

    def run_trial(self, trial_num):
        gaze = self.gaze_congruency[trial_num]
        preshape = self.hand_preshapes[trial_num]
        arm_vector = self.arm_vectors[trial_num]
        predicted_target = self.predict(gaze, preshape, arm_vector)
        actual_target = self.actual_target[trial_num]
        is_correct = int(predicted_target == actual_target)
        self.correct_predictions.append(is_correct)
        return is_correct
    

    def run_experiment(self):
        """Run the experiment and animate the results."""
        ani = FuncAnimation(self.fig, self.update, frames=range(self.trials),
                            init_func=self.init, blit=True, repeat=False)
        plt.show()
        self.generate_report()
    
    
    
    def update(self, frame):
        """Update the plot for each frame."""
        # Your existing logic for updating the plot...
        if frame < self.trials:  # Ensure we don't exceed the trial numbers
            is_correct = self.run_trial(frame)
            cumulative_accuracy = sum(self.correct_predictions) / (frame + 1) * 100
            self.accuracy_history.append(cumulative_accuracy)
        else:
            # Optional: You could explicitly call generate_report() here if you adjust for asynchronous behavior
            pass

        x_data = np.arange(0, len(self.accuracy_history))
        y_data = self.accuracy_history
        self.line.set_data(x_data, y_data)
        return self.line,