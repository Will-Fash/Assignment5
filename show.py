import matplotlib.pyplot as plt
import random


def showChart(x, y, title, range_, limit=5000):
    fig, ax1 = plt.subplots()
    c1 = ''
    c2 = ''
    c3 = ''
    c4 = ''
    c5 = ''
    c6 = ''
    c7 = ''
    c8 = ''
    c9 = ''
    c10 = ''
    c11 = ''
    c12 = ''
    c13 = ''
    c14 = ''
    comp1 = 0
    comp2 = 1
    # comp3 = 3

    colors = ['magenta', 'turquoise', 'darkorange', 'navajowhite', 'salmon'
        , 'azure', 'blue', 'brown', 'cadetblue', 'limegreen', 'maroon'
        , 'cornsilk', 'gold', 'black']

    for i in random.sample(range(range_), limit):
        if y[i] == 'bg':
            c1 = ax1.scatter(x[i, comp1], x[i, comp2], label='bg', color=colors[0])
        if y[i] == 'mk':
            c2 = ax1.scatter(x[i, comp1], x[i, comp2], label='mk', color=colors[1])
        if y[i] == 'bs':
            c3 = ax1.scatter(x[i, comp1], x[i, comp2], label='bs', color=colors[2])
        if y[i] == 'hr':
            c4 = ax1.scatter(x[i, comp1], x[i, comp2], label='hr', color=colors[3])
        if y[i] == 'sr':
            c5 = ax1.scatter(x[i, comp1], x[i, comp2], label='sr', color=colors[4])
        if y[i] == 'cz':
            c6 = ax1.scatter(x[i, comp1], x[i, comp2], label='cz', color=colors[5])
        if y[i] == 'sk':
            c7 = ax1.scatter(x[i, comp1], x[i, comp2], label='sk', color=colors[6])
        if y[i] == 'es-AR':
            c8 = ax1.scatter(x[i, comp1], x[i, comp2], label='es_AR', color=colors[7])
        if y[i] == 'es-ES':
            c9 = ax1.scatter(x[i, comp1], x[i, comp2], label='es_ES', color=colors[8])
        if y[i] == 'pt-BR':
            c10 = ax1.scatter(x[i, comp1], x[i, comp2], label='pt-BR', color=colors[9])
        if y[i] == 'pt-PT':
            c11 = ax1.scatter(x[i, comp1], x[i, comp2], label='pt-PT', color=colors[10])
        if y[i] == 'id':
            c12 = ax1.scatter(x[i, comp1], x[i, comp2], label='id', color=colors[11])
        if y[i] == 'my':
            c13 = ax1.scatter(x[i, comp1], x[i, comp2], label='my', color=colors[12])
        if y[i] == 'xx':
            c14 = ax1.scatter(x[i, comp1], x[i, comp2], label='xx', color=colors[13])

    scatters = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]

    ax1.legend(handles=scatters)
    ax1.grid(True)
    ax1.set_title(title)
    plt.show()