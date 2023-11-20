# model_factory.py
from tensorflow.keras import models, layers


class ConvBlockConfig:
    def __init__(self, filters, kernel_size, use_maxpooling=True, use_batchnorm=True, dropout_value=0, name=None):
        """Configuracion de cada bloque con los parametros que vienen del json

        Args:
            filters (_type_): Numero de filtros del bloque (una capa convolucional por bloque)
            kernel_size (_type_): Tamaño del kernel de la capa convolucional
            use_maxpooling (bool, optional): Usar regularizacion MaxPooling. Defaults to True.
            use_batchnorm (bool, optional): Usar regularizacion BatchNorm. Defaults to True.
            dropout_value (int, optional): Usar regularizacion Dropout. Defaults to 0.
            name (_type_, optional): Nombre del bloque. Defaults to None.
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_maxpooling = use_maxpooling
        self.use_batchnorm = use_batchnorm
        self.dropout_value = dropout_value
        self.name = name

def create_conv_block(config, input_shape):
    """Crea los bloques convolucionales con sus regularizaciones (MaxPooling, BatchNorm y Dropout)

    Args:
        config (_type_): Configuracion de cada bloque (viene del json de experimentos)
        input_shape (_type_): Tamaño de entrada del bloque

    Returns:
        _type_: Bloque convolucional
    """
    # Creacion del modelo
    block = models.Sequential(name=config.name)
    
    # CNN input 
    block.add(layers.Conv2D(config.filters, config.kernel_size, activation='relu', padding='same', input_shape=input_shape))
    
    # Regularizacion
    if config.use_batchnorm:
        block.add(layers.BatchNormalization(name=f'{config.name}_batchnorm'))
    
    if config.use_maxpooling:
        block.add(layers.MaxPooling2D((2, 2), name=f'{config.name}_maxpooling'))
    
    if config.dropout_value > 0:
        block.add(layers.Dropout(config.dropout_value))
    
    return block

def create_model_with_configurations(conv_block_configs, input_shape, num_classes):
    """Crea el modelo en base a los bloques que le hemos pasado con la configuracion de los experimentos

    Args:
        conv_block_configs (_type_): Configuracion de los bloques
        input_shape (_type_): Dimensiones de entrada
        num_classes (_type_): Numeros de clases para la capa de salida

    Returns:
        _type_: Modelo
    """
    model = models.Sequential()
    
    # Incluimos la capa de entrada directamente en el modelo principal
    model.add(layers.InputLayer(input_shape=input_shape, name='input'))
    
    # Creacion de los bloques para la extraccion de caracteristicas
    for i, config in enumerate(conv_block_configs, start=1):
        block = create_conv_block(config, input_shape)
        model.add(block)
        input_shape = (input_shape[0]//2, input_shape[1]//2, config.filters)  # Ajustamos input_shape después de MaxPooling
    
    # Creacion del top model para obtener la clase
    model.add(layers.Flatten(name='top_model_flatten'))
    model.add(layers.Dense(128, activation='relu', name='top_model_dense_1'))
    #model.add(layers.Dropout(0.5, name='top_model_dropout')) # Opcional
    model.add(layers.Dense(num_classes, activation='softmax', name='top_model_output'))
    
    return model