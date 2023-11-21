from data_functions import *
from model_factory import ConvBlockConfig, create_model_with_configurations
from custom_callbacks import get_callbacks
import json
import numpy as np
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

def show_results(history_list):
    # FUNCION DE BACKUP PARA MOSTRAR TODOS LOS ENTRENAMIENTOS DE GOLPE PARA HACER LA COMPARACION FINAL
    # Establecer la semilla para la generación de colores aleatorios
    np.random.seed(42)

    # Número de modelos
    num_models = len(history_list)

    # Seleccionar un mapa de colores (colormap)
    cmap = matplotlib.colormaps["tab10"]

    # Visualizar métricas y pérdidas en una sola gráfica
    plt.figure(figsize=(12, 8))

    # Plot de las pérdidas
    for i, history in enumerate(history_list):
        colors = [cmap(i*2/num_models), cmap(i*2/num_models + 0.5)]  # Dos colores por modelo
        plt.plot(history.history['loss'], label=f'Model {i+1} - Training Loss', linestyle='--', color=colors[0])
        plt.plot(history.history['val_loss'], label=f'Model {i+1} - Validation Loss', linestyle='--', color=colors[1])

    # Plot de la precisión
    for i, history in enumerate(history_list):
        colors = [cmap(i*2/num_models), cmap(i*2/num_models + 0.5)]  # Dos colores por modelo
        plt.plot(history.history['acc'], label=f'Model {i+1} - Training Accuracy', linestyle='-', color=colors[0])
        plt.plot(history.history['val_acc'], label=f'Model {i+1} - Validation Accuracy', linestyle='-', color=colors[1])

    plt.title('Model Training Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()

    # Añadir leyenda de líneas
    lines, labels = plt.gca().get_legend_handles_labels()
    labels = [label.replace('_', ' ').capitalize() for label in labels]  # Formatear leyenda
    plt.legend(lines, labels, loc='upper right')

    plt.grid(True)
    plt.show()


def load_data(gray_scale=False, augmenter=False, input_size=(224,224), batch_size=32):
    """Carga los tres datasets: train, test y valid

    Args:
        gray_scale (bool, optional): Dataset en escala de gris. Defaults to False.
        augmenter (bool, optional): Aumentar dataset en entrenamiento. Defaults to False.
        input_size (tuple, optional): Tamaño de entrada. Defaults to (224,224).
        batch_size (int, optional): Tamaño del batch. Defaults to 64.

    Returns:
        _type_: dataset
    """
    # Creates a dictionary mapping image paths to tuples containing class and partition.
    dict_dataset = get_dict_dataset()

    # Create dataframe
    df_dataset = dict2dataframe(dict_dataset)

    # Train
    df_dataset_train = df_dataset[df_dataset['split'] == 'Train'].drop(columns=['split'])

    # Valid
    df_dataset_valid = df_dataset[df_dataset['split'] == 'Valid'].drop(columns=['split'])

    # Test
    df_dataset_test = df_dataset[df_dataset['split'] == 'Test'].drop(columns=['split'])

    # Train dataset creation
    train_tfdataset = get_dataset(
        df=df_dataset_train,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        gray_scale=gray_scale,
        augmenter=augmenter,
        num_aug=1,
    )

    # Valid dataset creation
    valid_tfdataset = get_dataset(
        df=df_dataset_valid,
        input_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        gray_scale=gray_scale,
        augmenter=augmenter,
    )

    # Test dataset creation
    test_tfdataset = get_dataset(
        df=df_dataset_test,
        input_size=input_size,
        batch_size=batch_size,
        gray_scale=gray_scale,
        augmenter=augmenter,
        test_set=True,
    )


    # Calculamos los pesos de cada clase debido al desbalanceo de las clases
    computed_class_weights = compute_class_weight(class_weight='balanced', 
                                                  classes=np.unique(df_dataset_train['label'].values),
                                                  y=df_dataset_train['label'].values)
    class_weights = {idx: item for idx, item in enumerate(computed_class_weights)}

    # Num clases
    num_classes = df_dataset_train["label"].max() + 1

    return train_tfdataset, valid_tfdataset, test_tfdataset, class_weights, num_classes


def get_confusion_matrix(test_tfdataset, model_name, model):
    # EN PROGRESO HASTA QUE ARREGLEMOS LO DEL TEMA DE LAS LABELS DE TEST
    y_pred = []  # Predicted labels
    y_true = []  # GT Labels
    for image_batch, label_batch in test_tfdataset: 
        # GT Label
        y_true.append(label_batch)
        
        # Predicción
        preds = model.predict(image_batch)
        
        # Predicted labels
        y_pred.append(np.argmax(preds, axis = - 1))

    real_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)
    conf_matrix = confusion_matrix(real_labels, predicted_labels)

    # Crear un mapa de calor con seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicciones")
    plt.ylabel("Etiquetas Reales")
    plt.show()

    # Guardar la gráfica
    plt.savefig(f'models/{model_name}/confusion_matrix.png')
    plt.close()


def scratch_train(input_shape, experiments_json:str="scratch_experiments.json"):
    # Cargar datos
    train_tfdataset, valid_tfdataset, test_tfdataset, class_weights, num_classes = load_data()

    # Cargar configuraciones desde el archivo JSON
    with open(experiments_json, 'r') as file:
        experiment_config = json.load(file)

    # Crear modelo, compilarlo y entrenarlo
    history_list = []
    for experiment_name, config in experiment_config.items():
        print(f"Running Experiment: {experiment_name}")

        # Configurar el modelo utilizando las configuraciones
        block_configs = []
        for block_conf in config["model"]["conv_block_configs"]:
            block = ConvBlockConfig(block_conf["filters"], 
                                    block_conf["kernel_size"], 
                                    block_conf["use_maxpooling"],
                                    block_conf["use_batchnorm"],
                                    block_conf["dropout_value"],
                                    block_conf["name"])
            block_configs.append(block)
        
        # Crear modelo
        model = create_model_with_configurations(block_configs, input_shape, num_classes)
        
        # Compilar modelo
        model.compile(
            loss = tf.keras.losses.categorical_crossentropy,
            optimizer = tf.keras.optimizers.Adam(0.0001),
            metrics = ['acc']) 
                        #tfam.F1Score(num_classes=5, name='f1_weig', threshold=0.5)])
        
        # Crear callbacks
        callbacks = get_callbacks(experiment_name)

        # Mostrar arquitectura
        print(model.summary())

        # Entrenar modelo
        history = model.fit(train_tfdataset.repeat(),
                            validation_data=valid_tfdataset,
                            class_weight = class_weights,
                            epochs = config["training"]["epochs"],
                            steps_per_epoch=config["training"]["steps_per_epoch"],
                            callbacks = callbacks,
                            verbose=1)
        
        # get_confusion_matrix(test_tfdataset, experiment_name, model) # HAY QUE ARREGLARLO CON EL TEST

        history_list.append(history)


if __name__ == "__main__":
    # Dimensiones de entrada
    input_shape = (224, 224, 3)
    scratch_train(input_shape)