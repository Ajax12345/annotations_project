import matplotlib.pyplot as plt
import json, statistics


def plot_results() -> None:
    with open('model_results.json') as f:
        data = json.load(f)


    plt.plot([*range(1, len(data[-2][0])+1)], [statistics.mean(i) for i in zip(*data[-2])])
    plt.show()

def plot_bar() -> None:
    before_aug = {'Neither': 107, 'Bio': 17, 'About': 6, 'Product': 55, 'Title/role': 15}
    after_aug = {'Neither': 107, 'Bio': 50, 'About': 22, 'Product': 74, 'Title/role': 35}

    fig, [a2, a3] = plt.subplots(nrows=1, ncols=2)
    colors = ['red', 'blue', 'orange', 'green', 'purple']

    a2.bar([*before_aug], before_aug.values(), color = [f'tab:{i}' for i in colors])

    a3.bar([*after_aug], after_aug.values(), color = [f'tab:{i}' for i in colors])
    

    a2.title.set_text("Representation (Before Augmentation)")

    a2.set_ylabel("Frequency")

    a3.title.set_text("Representation (After Augmentation)")

    a3.set_ylabel("Frequency")

    plt.show()


if __name__ == '__main__':
    plot_bar()