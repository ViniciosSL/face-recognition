"""
FURB - Universidade Regional de Blumenau
Especialização em Data Science
Aprendizado de Máquina II - Aprendizado Não Supervisionado
Turma 2

> Reconhecimento facial utilizando PCA + kNN
    - Classe FaceImage

Autor: Marcus Moresco Boeno
Último update: 2020-12-05

"""

# Import de bibliotecas de terceiros
import numpy as np


class FaceImage:
    """Imagem da face de um ser humano

    > Parâmetros de Classe:
        - ??
    """
    # Método construtor
    def __init__(self, img_id:int, label:int, data:np.ndarray):
        self.img_id = img_id 
        self.label = label
        self.data = data
    
    @property
    def img_id(self) -> int:
        """Método Getter para ID da imagem

        > Arguments:
            - No arguments.
        
        > Output:
            - (int): ID da imagem.
        """
        return self.img_id_getter
    
    @img_id.setter
    def img_id(self, id_img:int):
        """Método Setter para ID da imagem

        > Arguments:
            - id_img (int): ID da imagem.
        
        > Output:
            - No output.
        """
        self.img_id_getter = id_img
    
    @property
    def label(self) -> int:
        """Método Getter para label da imagem

        > Arguments:
            - No arguments.
        
        > Output:
            - (int): label da imagem.
        """
        return self.label_getter
    
    @label.setter
    def label(self, img_label:int):
        """Método Setter para label da imagem

        > Arguments:
            - img_label (int): label da imagem.
        
        > Output:
            - No output.
        """
        self.label_getter = img_label
    
    @property
    def data(self) -> np.ndarray:
        """Método Getter para dados da imagem

        > Arguments:
            - No arguments.
        
        > Output:
            - (np.ndarray): Vetor coluna com dados da imagem.
        """
        return self.data_getter
    
    @data.setter
    def data(self, img_data:int):
        """Método Setter para dados da imagem

        > Arguments:
            - img_data (np.ndarray): Vetor coluna com dados da imagem.
        
        > Output:
            - No output.
        """
        self.data_getter = img_data

    def __str__(self):
        return f"{self.img_id} {self.label} {self.data}"


if __name__ == "__main__":

    test = FaceImage(2, 2, 1)
    print(test)
    print(test.label)
    test.label = 345
    print(test)
    