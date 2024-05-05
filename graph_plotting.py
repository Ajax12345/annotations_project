import matplotlib.pyplot as plt
import json, statistics


def plot_results() -> None:
    with open('model_results.json') as f:
        data = json.load(f)


    plt.plot([*range(1, len(data[-2][0])+1)], [statistics.mean(i) for i in zip(*data[-2])])
    plt.show()


if __name__ == '__main__':
    plot_results()