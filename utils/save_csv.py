import csv

def save_evaluation_results(filename, round_accuracies, round_losses):
    """
    Save evaluation results to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Accuracy (%)", "Loss"])
        for i, (acc, loss) in enumerate(zip(round_accuracies, round_losses)):
            writer.writerow([i + 1, acc, loss])
