import urllib.request
import zipfile
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class tensorflow():
    def __init__(self):
        self.url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
        self.DIR = "tmp/rps/"
        self.original_datagen = ImageDataGenerator(rescale=1./255)
        self.original_generator = self.original_datagen.flow_from_directory(self.DIR, 
                                                                batch_size=128, 
                                                                target_size=(150, 150), 
                                                                class_mode='categorical'
                                                                )
        
    def dataset_load(self):
        urllib.request.urlretrieve(self.url, 'rps.zip')
        self.local_zip = 'rps.zip'
        self.zip_ref = zipfile.ZipFile(self.local_zip, 'r')
        self.zip_ref.extractall('tmp/')
        self.zip_ref.close()

    def Image_Generator(self):
        self.training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest', 
        validation_split=0.2
        )
    
    def Make_Generator(self):
        self.training_generator = self.training_datagen.flow_from_directory(self.DIR, 
                                                          batch_size=32, 
                                                          target_size=(150, 150), 
                                                          class_mode='categorical', 
                                                          subset='training',
                                                         )
        self.validation_generator = self.training_datagen.flow_from_directory(self.DIR, 
                                                          batch_size=32, 
                                                          target_size=(150, 150), 
                                                          class_mode='categorical',
                                                          subset='validation', 
                                                         )
        
    def view_image(self):
        class_map = {
            0: 'Paper',
            1: 'Rock', 
            2: 'Scissors'
        }

        for x, y in self.original_generator:
            print(x.shape, y.shape)
            print(y[0])
            
            fig, axes = plt.subplots(2, 5)
            fig.set_size_inches(15, 6)
            for i in range(10):
                axes[i//5, i%5].imshow(x[i])
                axes[i//5, i%5].set_title(class_map[y[i].argmax()], fontsize=15)
                axes[i//5, i%5].axis('off')
            plt.gcf().canvas.set_window_title("original")
            plt.show()
            break
            
        for x, y in self.training_generator:
            print(x.shape, y.shape)
            print(y[0])
            
            fig, axes = plt.subplots(2, 5)
            fig.set_size_inches(15, 6)
            for i in range(10):
                axes[i//5, i%5].imshow(x[i])
                axes[i//5, i%5].set_title(class_map[y[i].argmax()], fontsize=15)
                axes[i//5, i%5].axis('off')
            plt.gcf().canvas.set_window_title("Augmentation")
            plt.show()
            break
        
    def makemodel(self):
        for x, y in self.original_generator:
            self.pic = x[:5]
            break
        
        conv2d = Conv2D(64, (3, 3), input_shape=(150, 150, 3))
        conv2d_activation = Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3))
        
        fig, axes = plt.subplots(8, 8)
        fig.set_size_inches(16, 16)
        for i in range(64):
            axes[i//8, i%8].imshow(conv2d(self.pic)[0,:,:,i], cmap='gray')
            axes[i//8, i%8].axis('off')
            
        fig, axes = plt.subplots(8, 8)
        fig.set_size_inches(16, 16)
        for i in range(64):
            axes[i//8, i%8].imshow(MaxPooling2D(2, 2)(conv2d(self.pic))[0, :, :, i], cmap='gray')
            axes[i//8, i%8].axis('off')
        conv1 = Conv2D(64, (3, 3), input_shape=(150, 150, 3))(self.pic)
        max1 = MaxPooling2D(2, 2)(conv1)
        conv2 = Conv2D(64, (3, 3))(max1)
        max2 = MaxPooling2D(2, 2)(conv2)
        conv3 = Conv2D(64, (3, 3))(max2)
        max3 = MaxPooling2D(2, 2)(conv3)
        
        #model
        model = Sequential([
            # Conv2D, MaxPooling2D 조합으로 층을 쌓습니다. 첫번째 입력층의 input_shape은 (150, 150, 3)으로 지정합니다.
            Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2), 
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2), 
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2), 
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2), 
            # 2D -> 1D로 변환을 위하여 Flatten 합니다.
            Flatten(), 
            # 과적합 방지를 위하여 Dropout을 적용합니다.
            Dropout(0.5),
            Dense(512, activation='relu'),
            # Classification을 위한 Softmax 
            # 출력층의 갯수는 클래스의 갯수와 동일하게 맞춰줍니다 (3개), activation도 잊지마세요!
            Dense(3, activation='softmax'),
        ])
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        checkpoint_path = "tmp_checkpoint.ckpt"
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    monitor='val_loss', 
                                    verbose=1)
        epochs=25
        history = model.fit(self.training_generator, 
                    validation_data=(self.validation_generator),
                    epochs=epochs,
                    callbacks=[checkpoint],
                    )
        model.load_weights(checkpoint_path)
        plt.figure(figsize=(12, 9))
        plt.plot(np.arange(1, epochs+1), history.history['acc'])
        plt.plot(np.arange(1, epochs+1), history.history['loss'])
        plt.title('Acc / Loss', fontsize=20)
        plt.xlabel('Epochs')
        plt.ylabel('Acc / Loss')
        plt.legend(['acc', 'loss'], fontsize=15)
        plt.show()
if __name__ == "__main__":
    tensor = tensorflow()
    
    #tensor.dataset_load()
    tensor.Image_Generator()
    tensor.Make_Generator()
    
    #tensor.veiw_image()
    tensor.makemodel()