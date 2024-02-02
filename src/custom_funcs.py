import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import scipy.stats as stats
import math
from sklearn.base import BaseEstimator, TransformerMixin


colors = ['#dfdfde', '#a2798f', '#d7c6cf', '#8caba8', '#ebdada']


def grafico_barras(feature, label):

    plt.figure(figsize=(6,4))

    # Criar um gráfico de contagem (histograma)
    ax = sns.countplot(
        x=feature,
        hue = feature,
        legend=False,
        palette=colors)

    # Adicionar rótulos aos topos das barras
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(patch.get_x() + patch.get_width() / 2, height + 0.1, str(int(height)), ha='center', va='bottom')
        
    # rótulo dos eixos
    plt.xlabel(label, fontsize=8)
    plt.ylabel('qtde', fontsize=8)

    # título do gráfico
    plt.title(f'Distribuição {label}', fontsize=12)

    # removendo as bordas
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Ajusta layout para evitar sobreposição
    plt.tight_layout()

    # Exibir o gráfico
    plt.show()


def grafico_distribuicao(feature):

    plt.hist(feature, bins=30, density=True, alpha=0.6, color='#d7c6cf')

    mu, sigma = np.mean(feature), np.std(feature)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    curva_normal = stats.norm.pdf(x, mu, sigma)

    plt.plot(x, curva_normal, 'r', color='#8caba8' ,label='Distribuição Normal')
    
    plt.xlabel('Valores')
    plt.ylabel('Densidade de Probabilidade')
    plt.title(f'Gráfico de Distribuição {feature.name} com Curva de Distribuição Normal')
    plt.legend()

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    plt.show()


def grafico_bloxpot(lista, data):

    a = math.ceil(len(lista)/2)

    fig, axes = plt.subplots(
        a,
        2,
        figsize=(10,(4*a))
    )

    for i, numeric in enumerate(lista):
        if i%2 == 0:
            x = 0
        else:
            x = 1

        if i < 2:
            y = 0
        else:
            y = math.floor(i/2)

        axes[y, x].boxplot(data[numeric], showmeans=True, medianprops=dict(color = '#a2798f'))
        axes[y, x].set_title(f'Boxplot {numeric}')

        axes[y, x].spines['top'].set_visible(False)
        axes[y, x].spines['right'].set_visible(False)

        max_value = np.max(data[numeric])

        axes[y, x].annotate(f'Valor Máximo = {max_value}',
                            xy=(1,max_value),
                            xytext = (1.05, max_value),
                            bbox = dict(facecolor='#d7c6cf', edgecolor='#a2798f'),
                            fontsize = 8
                            )


    if i%2 == 0:
        axes[math.floor(i/2), 1].axis('off')

    plt.tight_layout()

    plt.show()


def grafico_bloxpot_categorias(data, cat, num):
    fig = px.box(data,
                x=cat,
                y=num,
                color=cat,
                color_discrete_sequence=px.colors.qualitative.Set2)

    fig.update_layout(
        xaxis=dict(title=cat),
        yaxis=dict(title=num),
        title= num + " per " + cat,
        showlegend=True
    )

    fig.show()


def grafico_dispersao(data, x, y):

    sns.set(rc={"figure.figsize": (8, 6)})
    sns.scatterplot(
        x=x, 
        y=y, 
        data = data,
        color = '#d7c6cf')

    # título do gráfico
    plt.title(f'Gráfico de dispersão {y} x {x}', fontsize=12)

    # removendo as bordas
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Ajusta layout para evitar sobreposição
    plt.tight_layout()

    # Exibir o gráfico  
    plt.show()
  

def feature_importance(model, X_train):

    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    model_name = type(model).__name__

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(feature_importances)), feature_importances, align='center')
    plt.yticks(np.arange(len(feature_names)), feature_names)

    plt.xlabel('Importância das Variáveis')
    plt.title(f'Importância das Variáveis no {model_name}')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, categories_dict, columns):
         self.categories_dict = categories_dict
         self.columns = columns

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns = self.columns).copy()
        
        for feature, classes in self.categories_dict.items():
            for n in range(len(classes)):
                X[feature].replace(classes[n], n, inplace=True)
            
            X[feature] = X[feature].apply(lambda item: -1 if isinstance(item, str) else item)

        return X
    

class NullToBinary(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature):
         self.feature = feature

    def fit(self, X, y = None):
        return self

    def transform(self, X):                 
        X[self.feature] = X[self.feature].apply(lambda item: 0 if pd.isna(item) else 1)

        return X
    

def RemoveOutliers(X, colunas, n):
    for feat in colunas:
        media = X[feat].mean()
        desvio_padrao = X[feat].std()

        limite_superior = media + n * desvio_padrao
        limite_inferior = media - n * desvio_padrao

        X.drop(X[X[feat] > limite_superior].index, inplace=True)
        X.drop(X[X[feat] < limite_inferior].index, inplace=True)


