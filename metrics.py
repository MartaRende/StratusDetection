import numpy as np
import matplotlib.pyplot as plt
import os
class Metrics:
    def __init__(self, expected, predicted, data, save_path=None):
        self.expected = expected
        self.predicted = predicted
        self.data = data['dole']
        self.save_path = save_path
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Create a subfolder for metrics (e.g., "metrics")
            metrics_folder = os.path.join(save_path, "metrics")
            os.makedirs(metrics_folder, exist_ok=True)

            # Update save_path to point to the new folder
            self.save_path = metrics_folder

    
    def get_correctPredictions(self,expected, predicted, tol=20):
        good_predictions = 0
        for i in range(len(expected)):
            good_predictions += int(np.all(np.abs(predicted[i] - expected[i]) <= tol))
        return good_predictions
    def get_accuracy(self,expected, predicted, tol=20):
        good_predictions = self.get_correctPredictions(expected, predicted, tol)
        accuracy = good_predictions / len(expected)
        return accuracy
    def find_datetime(self, expected, predicted):
        matching_items = [
            (i, item)
            for (i, item) in enumerate(self.data)
            if abs(item["gre000z0_dole"] - expected[1]) <= 1 # for approximation
            and abs(item["gre000z0_nyon"] - expected[0]) <= 1
        ]
        index = matching_items[0][0] if matching_items else None
        if index is not None:
            datetime = self.data[index]["datetime"]
            return datetime
        else:
            print("No matching items found for expected values:", expected)
            return []
    def print_datetimes(self):
        for i in range(len(self.expected)):
            datetime = self.find_datetime(self.expected[i], self.predicted[i])
            if datetime:
                print("Datetime:", datetime, "Predicted values:", self.predicted[i], "Expected values:", self.expected[i])
            else:
                print(f"Sample {i}: No matching datetime found")
    def find_datetimes(self):
        datetime_list = []
        for i in range(len(self.expected)):
            datetime_list.append(self.find_datetime(self.expected[i], self.predicted[i]))
        return datetime_list
    def get_mean_absolute_error(self):
        mae= []
        
        for i in range(len(self.expected)):
            mae1 = np.mean(np.abs(self.expected[i][0] - self.predicted[i][0]))
            mae2 = np.mean(np.abs(self.expected[i][1] - self.predicted[i][1]))
            
            mae.append([mae1, mae2])
        return mae
    
    def plot_mae(self, mea, title="Mean Absolute Error", xlabel="Datetime", ylabel="MAE"):
        plt.figure(figsize=(12, 10))
        datetime = self.find_datetimes()
        
        if isinstance(mea[0], (list, np.ndarray)):
            for i, dataset in enumerate(zip(*mea)):
                label = "nyon" if i == 0 else "dole"
                plt.plot(dataset, marker='o', linestyle='-', label=label)

            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)
        else:
            plt.plot(mea, marker='o', linestyle='-', color='b', label='MAE')
            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.grid(True)
        plt.legend()

        # save
        plt.savefig(f"{self.save_path}/mae.png")
        
    def get_delta_between_expected_and_predicted(self):
        delta = []
        for i in range(len(self.expected)):
            delta1 = np.abs(self.expected[i][0] - self.predicted[i][0])
            delta2 = np.abs(self.expected[i][1] - self.predicted[i][1])
            delta.append([delta1, delta2])
        return delta
    
    def plot_delta(self, delta, title="Delta between Expected and Predicted", xlabel="Datetime", ylabel="Delta"):


        plt.figure(figsize=(12, 10))
        datetime = self.find_datetimes()

        # Check if delta[0] contains multiple datasets
        if isinstance(delta[0], (list, np.ndarray)):
            for i, dataset in enumerate(zip(*delta)):  # Unpack multiple datasets
                label = "nyon" if i == 0 else "dole"
                plt.plot(dataset, marker='o', linestyle='-', label=label)

            # Show only every 6th datetime label for clarity
            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)
        else:
            plt.plot(delta, marker='o', linestyle='-', color='b', label='Delta')
            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.legend()
        plt.subplots_adjust(bottom=0.25)  # adjust space for rotated labels
        plt.grid(True)
    
        plt.savefig(f"{self.save_path}/delta.png")
    def get_relative_error(self):
        relative_error = []
        for i in range(len(self.expected)):
            rel_error1 = np.abs((self.expected[i][0] - self.predicted[i][0]) / self.expected[i][0])
            rel_error2 = np.abs((self.expected[i][1] - self.predicted[i][1]) / self.expected[i][1])
            relative_error.append([rel_error1, rel_error2])
        return relative_error
    def plot_relative_error(self, relative_error, title="Relative Error", xlabel="Datetime", ylabel="Relative Error"):
        plt.figure(figsize=(12, 10))
        datetime = self.find_datetimes()

        if isinstance(relative_error[0], (list, np.ndarray)):
            for i, dataset in enumerate(zip(*relative_error)):
                label = "nyon" if i == 0 else "dole"
                plt.plot(dataset, marker='o', linestyle='-', label=label)

            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)
        else:
            plt.plot(relative_error, marker='o', linestyle='-', color='b', label='Relative Error')
            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_path}/relative_error.png")
        plt.close()

    def get_mse(self):
        mse = []
        for i in range(len(self.expected)):
            mse1 = np.mean((self.expected[i][0] - self.predicted[i][0]) ** 2)
            mse2 = np.mean((self.expected[i][1] - self.predicted[i][1]) ** 2)
            mse.append([mse1, mse2])
        return mse
    def plot_mse(self, mse, title="Mean Squared Error", xlabel="Datetime", ylabel="MSE"):
        plt.figure(figsize=(12, 10))
        datetime = self.find_datetimes()

        if isinstance(mse[0], (list, np.ndarray)):
            for i, dataset in enumerate(zip(*mse)):
                label = "nyon" if i == 0 else "dole"
                plt.plot(dataset, marker='o', linestyle='-', label=label)

            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)
        else:
            plt.plot(mse, marker='o', linestyle='-', color='b', label='MSE')
            xtick_labels = [dt if idx % 6 == 0 else "" for idx, dt in enumerate(datetime)]
            plt.xticks(range(len(datetime)), xtick_labels, rotation=90)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_path}/mse.png")
        plt.close()
