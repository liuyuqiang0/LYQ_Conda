import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import wget,zipfile,os


# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Cat_Vs_Dog(object):
    def __init__(self):
        self.url='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        self.base_dir='.' # 指定所有工作在当前工作目录下

    def Get_Data(self):
        if 'cats_and_dogs_filtered' in os.listdir('.'): # 已经下载过了
            self.base_dir = './cats_and_dogs_filtered'
            return
        print('Downloading...')
        wget.download(self.url)
        local_data='cats_and_dogs_filtered.zip'
        zip_ref=zipfile.ZipFile(local_data,'r')
        zip_ref.extractall(self.base_dir) # 没有指定解压目录就自动解压到当前工作目录 （注：此数据集已经分好了各级目录）
        self.base_dir = './cats_and_dogs_filtered'
        zip_ref.close()

    def Directory_Define(self):
        train_dir=os.path.join(self.base_dir,'train') # 训练目录，图像生成器(数据预处理)传入这个参数
        validation_dir = os.path.join(self.base_dir, 'validation')

        train_cats_dir=os.path.join(train_dir,'cats')
        train_dogs_dir=os.path.join(train_dir,'dogs')

        validation_cats_dir=os.path.join(validation_dir,'cats')
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        return train_dir,validation_dir,train_cats_dir,train_dogs_dir,validation_cats_dir,validation_dogs_dir

    def Show_Some_Image(self,train_cats_dir,train_dogs_dir,validation_cats_dir,validation_dogs_dir):
        train_cat_fnames = os.listdir(train_cats_dir)
        train_dog_fnames = os.listdir(train_dogs_dir)
        print(train_cat_fnames[:10])
        print(train_dog_fnames[:10])

        print('total training cat images :', len(os.listdir(train_cats_dir))) # 查看训练集大小和测试集大小方便后面设置图片生成器批次大小
        print('total training dog images :', len(os.listdir(train_dogs_dir)))

        print('total validation cat images :', len(os.listdir(validation_cats_dir)))
        print('total validation dog images :', len(os.listdir(validation_dogs_dir)))

        nrows,ncols=4,4
        next_cat_pix=[os.path.join(train_cats_dir,fname) for fname in train_cat_fnames[:8]]  # 打印训练集中猫和狗图片各8张
        next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[:8]]
        for i,img_path in enumerate(next_cat_pix+next_dog_pix):
            plt.subplot(nrows,ncols,i+1)
            plt.axis('Off')
            img=mpimg.imread(img_path)   # 需要pillow包的支持，不然只能读.png格式的图片
            plt.imshow(img)
        plt.show()

    def Data_Preprocess(self,train_dir,validation_dir):
        train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        train_generator=train_datagen.flow_from_directory(
            train_dir,
            target_size=(150,150),
            batch_size=100,
            class_mode='binary'
        )
        validation_datagen=ImageDataGenerator(rescale=1.0/255)
        validation_generator=validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(150,150),
            batch_size=50,
            class_mode='binary'
        )
        return train_generator,validation_generator

    def Build_CNN_Model(self):
        model=tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
            tf.keras.layers.MaxPool2D(2,2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ])
        return model

    def Compile_Trainning(self,model,train_generator,validation_generator,epochs):
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=0.001), # 学习率
                      metrics=['acc'])

        history=model.fit_generator(
            train_generator,
            steps_per_epoch=20,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=20,
            verbose=2
        )
        # self.Save_Load_Model('./model_0.h5',1,model)
        return model,history

    def Save_Load_Model(self,path,Is_Save=1,model=None): # 默认保存模型
        if Is_Save:
            model.save(path)
        else:
            return tf.keras.models.load_model(path)

    def Evaluating_Acc_and_Loss(self,history):
        train_acc=history.history['acc']
        validation_acc=history.history['val_acc']
        train_loss=history.history['loss']
        validation_loss=history.history['val_loss']
        plt.subplot(1,2,1)
        epochs = range(len(train_acc))  # Get number of epochs
        plt.plot(epochs,train_acc,label='train_acc')
        plt.plot(epochs,validation_acc,label='validation_acc')
        plt.legend(loc='best')
        plt.title ('Training and validation accuracy')

        plt.subplot(1,2,2)
        plt.plot(epochs, train_loss,label='train_loss')
        plt.plot(epochs, validation_loss,label='validation_loss')
        plt.legend(loc='best')
        plt.title('Training and validation loss')
        plt.show()

if __name__=='__main__':
    Is_Train = True
    Is_Save, model_path = 1 , './model_1.h5'
    AI = Cat_Vs_Dog()  # 实例化

    if Is_Train:  # 构建模型并保存
        AI.Get_Data() # 下载数据集
        train_dir,validation_dir,train_cats_dir,train_dogs_dir,validation_cats_dir,validation_dogs_dir=AI.Directory_Define() # 定义训练目录与验证目录
        AI.Show_Some_Image(train_cats_dir,train_dogs_dir,validation_cats_dir,validation_dogs_dir)
        train_generator,validation_generator=AI.Data_Preprocess(train_dir,validation_dir) # 图像生成器
        model=AI.Build_CNN_Model() # 构建卷积模型
        model.summary() # 图层概览
        model,history=AI.Compile_Trainning(model,train_generator,validation_generator,epochs=15) # 模型编译与训练
        AI.Evaluating_Acc_and_Loss(history)
        if Is_Save:
            AI.Save_Load_Model(model_path, Is_Save,model) #  保存模型(注意传入的是model而不是history，history只是训练结果

    else: # 加载模型进一步操作
        model=AI.Save_Load_Model(model_path,Is_Save)
        # AI.Evaluating_Acc_and_Loss(model)
        model.summary()









