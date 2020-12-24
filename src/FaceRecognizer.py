"""
FURB - Universidade Regional de Blumenau
Especialização em Data Science
Aprendizado de Máquina II - Aprendizado Não Supervisionado
Turma 2

> Reconhecimento facial utilizando PCA + kNN
    - Classe FaceRecognizer

Autor: Marcus Moresco Boeno
Último update: 2020-12-24

"""

# Imports de bibliotecas built-in
from math import sqrt

# Import de bibliotecas de terceiros
from matplotlib.image import imread, imsave
import numpy as np


class FaceRecognizer:
    """Implementação de reconhecimento facial utilizando PCA e kNN

    > Parâmetros de Classe:
        - img_list (list): Lista de imagens da classe FaceImage
    """
    
    # Método construtor
    def __init__(self, img_list):
        self.__imgs = img_list
    
    @property
    def imgs(self) -> list:
        """Método Getter para lista de imgs

        > Arguments:
            - No arguments.
        
        > Output:
            - (list): Lista de objetos da classe FaceImage.
        """
        return self.__imgs_getter
    
    @imgs.setter
    def imgs(self, img_list:str):
        """Método Setter para lista de imgs

        > Arguments:
            - img_list (list): Lista de objetos da classe FaceImage.
        
        > Output:
            - No output.
        """
        self.__imgs_getter = img_list
    
    def fit(self, k:int):
        """Ajusta o modelo de acordo com o número k de componentes

        > Argumentos:
            - k (int): Número de componentes (eigenfaces) a serem
                utilizadas para o treinamento do modelo
        
        > Output:
            - Sem output.
        """
        # Calcula imagem média
        self.__calcMean()
        
        # Calcula matriz com diferenças
        self.__calcDiff()

        # Calcula matriz de covariancia
        cov_mat = np.matmul(self.__diffs.T, self.__diffs)

        # Calcula autovalores e autovetores da matriz de covariancia
        evls, evts = np.linalg.eig(cov_mat)
        
        # Ordena autovalores e autovetores por autovalor decrescente
        evts_sorted = [y for _,y in sorted(zip(evls, evts), reverse=True)]

        # Calcula eigenfaces
        self.__calcEigenFaces(evts_sorted, k)

        # # Exporta eigenfaces
        # self.__save_eigen_faces()

        # Projeta imagens de treino para novo espaço k-dimensional
        self.__projections = np.matmul(self.__eigenfaces.T, self.__diffs)
    
    def __save_eigen_faces(self):
        """Exporta k-eigenfaces
        
        > Argumentos:
            - Sem argumentos.
        
        > Output:
            - Sem output.
        """
        for pos, eigenface in enumerate(self.__eigenfaces.T):
            imsave(
                "".join(["eigenface_", str(pos+1), ".jpg"]), 
                eigenface.reshape(70, 80).T
                )
    
    def __calcEigenFaces(self, evts:list, k:int):
        """Calcula eigenfaces
        
        > Argumentos:
            - evts (list): Lista contendo autovalores e autovetores;
            - k (int): Número de componentes para cálculo.
        
        > Output:
            - Sem output.
        """
        # Monta matriz com autovetores de interesse
        k_evts = evts[0].reshape(evts[0].size, 1)
        for j in range(1, k):
            k_evts = np.hstack((k_evts, evts[j].reshape(evts[j].size, 1)))
        
        # Calcula eigenfaces
        eigenfaces = np.matmul(self.__diffs, k_evts)

        # Aplica normalização L2 em cada eigenface
        eigenfaces = np.apply_along_axis(
            # Função anônima (lambda) para cálculo da normalização L2
            func1d=lambda x: np.divide(x, sqrt(np.sum(np.power(x, 2)))),
            axis=0,
            arr=eigenfaces.T
        ).T

        # Salva eigenfaces como atributo da instância
        self.__eigenfaces = eigenfaces

    def __calcMean(self):
        """Calcula imagem média

        > Argumentos:
            - Sem argumentos.
        
        > Output:
            - Sem output.        
        """
        # Le dados da primeira imagem para formar base da imagem média
        mean_img = np.array(imread(self.__imgs[0].data), np.float64)
        
        # Calcula imagem média
        for i in range(1, len(self.__imgs)):
            mean_img += np.array(imread(self.__imgs[i].data), np.float64)
        mean_img = mean_img/len(self.__imgs)

        # Salva imagem média como atributo da instância
        self.__mean_img = mean_img.T.reshape(mean_img.size, 1)
    
    def __calcDiff(self):
        """Calcula matriz com diferenças entre imagem média e imagens de
        treino

        > Argumentos:
            - Sem argumentos
        
        > Output:
            - Sem output
        """
        # Inicia array de diffs com a primeira imagem
        sample = imread(self.__imgs[0].data)
        diffs = np.subtract(sample.T.reshape(sample.size, 1), self.__mean_img)

        # Calcula diferenças entre imagem média e imagens de treino restantes
        for i in range(1, len(self.__imgs)):
            tmp = imread(self.__imgs[i].data)
            tmp = np.subtract(tmp.T.reshape(tmp.size, 1), self.__mean_img)
            diffs = np.hstack((diffs, tmp))
        
        # Salva diffs como atributo da instância
        self.__diffs = diffs

    def predict(self, imgs_predict:list) -> list:
        """Classifica uma imagem com o modelo treinado

        > Argumentos:
            - imgs_predict (list): Lista de imagens da classe FaceImage 
                a serem classificadas.
        
        > Ouput:
            - (list): Lista contendo tuplas com 2 elementos:
                [0]: Classificação (label) da imagem
                [1]: Distância ao vizinho mais próximo
        """
        # Cria lista vazia para armazenar os resultados
        res = []

        # Recupera labels das imagens de treino
        labels = [img.label for img in self.__imgs]
                
        # Itera sobre imagens a serem classificadas
        for img in imgs_predict:

            img_class = [img.label]
            
            # Calcula diferença entre imagem média e imagem a ser classificada
            img_data = imread(img.data)
            diff = np.subtract(
                img_data.T.reshape(img_data.size, 1), 
                self.__mean_img
            )

            # Projeta a nova imagem para o espaço k-dimensional das eigenfaces
            img_projected = np.matmul(self.__eigenfaces.T, diff)

            # Calcula distancia euclidiana entre nova imagem e imagens de treino
            dist = np.sum(
                np.square(img_projected - self.__projections), 
                axis = 0
            )
                            
            # Recupera label mais próxima e adiciona ao vetor com respostas
            res.append([[y, x] for x, y in sorted(zip(dist, labels))][0])
        
        # Retorna labels e distâncias
        return res
