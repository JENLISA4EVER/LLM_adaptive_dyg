import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from typing import Dict, List

class RealtimePlotter:
    """
    A class to handle real-time plotting and saving of evaluation metrics.
    It is designed to be robust against interruptions.
    """
    def __init__(self, task_name: str, save_path: str, metrics: List[str]):
        """
        Initializes the plotter.

        Args:
            task_name (str): The name of the task (e.g., 'Link Prediction', 'Node Retrieval').
            save_path (str): The file path to save the plot image (e.g., 'link_prediction_metrics.png').
            metrics (List[str]): A list of metric names to be plotted (e.g., ['ROC AUC', 'AP']).
        """
        self.task_name = task_name
        self.save_path = save_path
        self.metrics_to_plot = metrics
        self.history = []  # List to store dictionaries of {'event_count': count, 'metric': name, 'value': val}
        
        # Set a professional plot style
        sns.set_theme(style="whitegrid")
        print(f"RealtimePlotter initialized for '{task_name}'. Plots will be saved to '{save_path}'.")

    def update_and_plot(self, event_count: int, current_metrics: Dict[str, float]):
        """
        Updates the history with new metrics and generates/saves a new plot.
        This function is called periodically.
        """
        # Add new data points to history
        for metric_name, value in current_metrics.items():
            if metric_name in self.metrics_to_plot:
                self.history.append({
                    'Test Events Processed': event_count,
                    'Metric': metric_name,
                    'Value': value
                })

        if not self.history:
            return

        # Create a DataFrame for easy plotting with seaborn
        df = pd.DataFrame(self.history)

        # Create the plot
        plt.figure(figsize=(12, 7))
        
        plot = sns.lineplot(
            data=df,
            x='Test Events Processed',
            y='Value',
            hue='Metric',
            style='Metric',
            markers=True,
            dashes=False
        )
        
        # Formatting the plot for better readability
        plot.set_title(f'Performance on {self.task_name} Task over Time', fontsize=16, weight='bold')
        plot.set_xlabel('Number of Test Events Processed', fontsize=12)
        plot.set_ylabel('Metric Value', fontsize=12)
        plot.legend(title='Metrics')
        plot.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Ensure x-axis ticks are integers
        plot.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.setp(plot.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()

        # Save the figure, overwriting the previous one.
        # This is an atomic operation on most filesystems, making it robust.
        try:
            plt.savefig(self.save_path, dpi=300)
        except Exception as e:
            print(f"Error saving plot: {e}")
        
        # Close the figure to free up memory
        plt.close()