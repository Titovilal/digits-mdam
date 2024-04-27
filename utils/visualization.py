import matplotlib.pyplot as plt
import seaborn as sns  # Importa la librería Seaborn para obtener una paleta de colores
def plot_MLP(mlp_accuracys):
    plt.plot(range(1, len(mlp_accuracys)+1), mlp_accuracys, label='MLP Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel("Repetition number")
    plt.show()

def plot_KNN_scores_combined(knn_cross_values_list):
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(knn_cross_values_list))

    for i, knn_cross_values in enumerate(knn_cross_values_list):
        plt.plot(range(1, len(knn_cross_values) + 1), knn_cross_values, label=f'Repetition {i + 1}', color=colors[i])

    # Calcula la media de los valores de F1 para cada valor de k
    mean_f1_scores = [sum(scores) / len(scores) for scores in zip(*knn_cross_values_list)]
    plt.plot(range(1, len(mean_f1_scores) + 1), mean_f1_scores, label='Media', linestyle='--', color='black')

    plt.legend(loc="lower right")
    plt.xlabel("K")
    plt.ylabel("F1 score")
    plt.title("Combined KNN F1 Scores")
    plt.show()
