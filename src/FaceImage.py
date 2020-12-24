"""
FURB - Universidade Regional de Blumenau
Especialização em Data Science
Aprendizado de Máquina II - Aprendizado Não Supervisionado
Turma 2

> Reconhecimento facial utilizando PCA + kNN
    - Classe FaceImage

Autor: Marcus Moresco Boeno
Último update: 2020-12-24

"""

class FaceImage:
    """Imagem da face de um ser humano

    > Parâmetros de Classe:
        - img_id (int): ID da imagem;
        - label (int): Label da imagem;
        - data (str): Caminho (system path) até a imagem.
    """
    # Método construtor
    def __init__(self, img_id:int, label:int, data:str):
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
    def data(self) -> str:
        """Método Getter para dados da imagem

        > Arguments:
            - No arguments.
        
        > Output:
            - (str): Caminho (system path) até a imagem.
        """
        return self.data_getter
    
    @data.setter
    def data(self, img_data:str):
        """Método Setter para dados da imagem

        > Arguments:
            - img_data (str): Caminho (system path) até a imagem.
        
        > Output:
            - No output.
        """
        self.data_getter = img_data

    def __str__(self):
        return f"ID={self.img_id}; Label={self.label}; Path={self.data}"
