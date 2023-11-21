from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class HistoryAndPlotCallback(Callback):
    def __init__(self, model_name):
        super(HistoryAndPlotCallback, self).__init__()
        self.model_name = model_name

    def on_train_end(self, logs=None):
        # Guardar el historial del modelo
        history = self.model.history.history

        # Calcular límite eje Y
        loss_av = np.average(history["loss"][-1])
        val_loss_av = np.average(history["val_loss"][-1])
        acc_av = np.average(history["acc"][-1])
        val_acc_av = np.average(history["val_acc"][-1])
        y_upper_limit = np.max([loss_av, val_loss_av, acc_av, val_acc_av])
        y_upper_limit *= 2

        np.save(f'models/{self.model_name}/training_history.npy', history)

        # Guardar métricas y pérdidas en una sola gráfica
        epochs = len(history["loss"])
        
        plt.style.use("ggplot")
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(0, epochs), history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), history["acc"], label="train_acc")
        plt.plot(np.arange(0, epochs), history["val_acc"], label="val_acc")
        
        plt.title(f"Training Loss and Accuracy - {self.model_name}")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc='upper left')
        plt.ylim(0, int(y_upper_limit))
        plt.grid(True)
        
        # Guardar la gráfica
        plt.savefig(f'models/{self.model_name}/training_plot.png')
        plt.close()

        # Guardar history para revisar en el futuro por si se activa un earlystopping o cualquier problema
        pd.DataFrame.from_dict(history).to_csv(f'models/{self.model_name}/history.csv', index=False)


def get_callbacks(model_name):
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss', 
            mode = 'min', 
            patience = 20, 
            verbose=1,
            restore_best_weights=False),
        ModelCheckpoint(
            filepath = f'models/{model_name}',
            monitor = 'val_loss',
            mode = 'min',
            save_best_only = True,
            verbose = 1),
        HistoryAndPlotCallback(model_name),

    ]
        
    return callbacks