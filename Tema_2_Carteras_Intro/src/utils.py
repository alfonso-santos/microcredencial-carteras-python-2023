import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def dibuja_covar(data, titulo = 'Matriz de Covarianzas'):
    """
    Dibuja la matriz de covarianzas de un conjunto de datos
    :param data: Matriz de covarianzas
    :param titulo: Título del gráfico
    :return: 0
    """
    
        # Crear un gráfico de matriz de covarianzas
    plt.figure(figsize=(8, 6))

    vmin = -1
    vmax = 1
    plt.imshow(data, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

    # Mostrar los valores de covarianza en cada celda
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data.iloc[i, j]:.2f}', va='center', ha='center', color='black', fontsize=14)

    # Agregar límites de celdas con líneas negras
    for i in range(data.shape[0] + 1):
        plt.axhline(i - 0.5, color='black', linewidth=1)
        plt.axvline(i - 0.5, color='black', linewidth=1)

        # Agregar un cuadrado exterior
    outer_rect = patches.Rectangle((-0.5, -0.5), data.shape[0], data.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(outer_rect)

    # Añadir una barra de colores
    plt.colorbar(label='Covariance')
    plt.title(titulo)
    # plt.xticks(np.arange(data.shape[0])), np.arange(1, data.shape[0] + 1)
    # plt.yticks(np.arange(data.shape[1])), np.arange(1, data.shape[1] + 1)
    plt.xticks(np.arange(data.shape[0]), data.columns)
    plt.yticks(np.arange(data.shape[1]), data.columns)

    # Desactivar las líneas horizontales y verticales
    plt.grid(False)

    plt.show()

    return 0


def dibuja_covar_ax(data, ax):
    """
    Dibuja la matriz de covarianzas de un conjunto de datos en una figura con subgráficos
    :param data: Matriz de covarianzas
    :param ax: Subgráfico donde se dibujará la matriz de covarianzas
    :return: 0
    """ 
    # Crear un gráfico de matriz de covarianzas en el subgráfico especificado por 'ax'
    vmin = 0
    vmax = 1
    im = ax.imshow(data, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

    # Mostrar los valores de covarianza en cada celda
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]:.2f}', va='center', ha='center', color='white')

    # Agregar límites de celdas con líneas negras
    for i in range(data.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    # Agregar un cuadrado exterior
    outer_rect = patches.Rectangle((-0.5, -0.5), data.shape[0], data.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(outer_rect)

    # Añadir una barra de colores
    plt.colorbar(im, ax=ax, label='Covariance')
    ax.set_title('Matriz de Covarianzas')
    plt.xticks(np.arange(data.shape[0])), np.arange(1, data.shape[0] + 1)
    plt.yticks(np.arange(data.shape[1])), np.arange(1, data.shape[1] + 1)  
