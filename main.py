#Библиотеки для работы с данными
import numpy as np
#Для работы с моделями
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import keras
import tensorflow as tf
#Для работы с изображениями
import cv2
from PIL import Image
#Для работы с метрикой f1-score
import tensorflow_addons as tfa
f1 =tfa.metrics.F1Score(num_classes=9, average='weighted')

#Класс для получения предсказания на основании сохраненных моделей:
class EmoRecog():
    # Названия классов:
    em_names = {0: 'anger',
         1: 'contempt',
         2: 'disgust',
         3: 'fear',
         4: 'happy',
         5: 'neutral',
         6: 'sad',
         7: 'surprise',
         8: 'uncertain'}
    
    def __init__(self):
         self.model_acc = tf.keras.models.load_model(".../checkpoints/emotion_recog/acc_2blocks.h5", compile=False)
         self.model_f1 = tf.keras.models.load_model(".../checkpoints/emotion_recog/f1_2blocks.h5", compile=False)
         #Создадим объект класса ImageDataGenerator (для подачи в модели):
         self.test_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)
    def predict(self, img):
        #Подаем в модель
        test_generator = self.test_datagen.flow(img, batch_size = 1, seed =12, shuffle  = False)
        #Получаем предсказания:
        acc_preds = self.model_acc.predict(test_generator)
        f1_preds = self.model_f1.predict(test_generator)
        model_preds = np.array([acc_preds, f1_preds])
        model_preds = np.tensordot(model_preds, [0.91803279, 0.08196721], axes=((0),(0)))
        model_preds = EmoRecog.em_names[np.argmax(model_preds)]
        return model_preds


#Класс для работы с видео:
class VideoDisplay():
    #Шрифт
    font = cv2.FONT_HERSHEY_SIMPLEX
    #Размер шрифта
    fontScale = 0.6
    # Толщина линии (пикселей)
    thickness = 1
    #Цвет (зеленый)
    font_color = (0, 255, 0)
    #Параметры фрейма:
    xmax=640 
    ymax=480
    #Для изменения размеров далее
    required_size = (224, 224)
    
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        self.vid.set(3, 448)  # установка ширины дисплея
        self.vid.set(4, 224)  # установка высоты дисплея
        #Проверка подключения камеры:
        if not (self.vid.isOpened()):
            print("Could not open video device")
        self.face_detection_model = cv2.FaceDetectorYN_create('.../data/YuNet/face_detection_yunet_2023mar.onnx',
                          "", 
                          (self.xmax, self.ymax),                
                          score_threshold=0.5)

    def __del__(self):
        self.vid.release()
        
    def putText (self, frame, text, x,y):
        cv2.putText(frame, text, (x, y), self.font, self.fontScale, self.font_color, self.thickness, cv2.LINE_AA)
        cv2.imshow('camera', frame)

    def noface_error(self):
        #Если лица не находятся, выводим сообщение на кадр:
        #Напечатаем текст для проверки
        print('No faces detected!')
        self.putText('No faces detected!', 50, 50)

    def face_to_array(self, frame, face_boundary):
        face_boundary = Image.fromarray(face_boundary)
        face_image = face_boundary.resize(self.required_size)
        face_array = np.asarray(face_image,'float32')
        face_array = face_array[None, ...]
        return face_array
        

    def pred_emotion(self, model):
        # Разделяем видео на отдельные кадры-фреймы
        ret, frame = self.vid.read()
        #Находим лица
        faces = self.face_detection_model.detect(frame)[1]


        if faces is not None:
            for face in faces:
                face = face.astype(int)
                x1, y1, f_width, f_height = face[0], face[1], face[2], face[3]
                x2, y2 = x1 + f_width, y1 + f_height


                # Следим, чтобы бокс не вылез за пределы экрана, иначе функция вылетит с ошибкой:
                if x2 > self.xmax:
                    x2 = self.xmax
                if y2 > self.ymax:
                    y2 = self.ymax
        
                # Нарисуем бокс вокруг лица
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face_boundary = frame[y1:y2, x1:x2]
                # обрезка изображения до рамки лица для подачи в модель
                face_array = self.face_to_array(frame, face_boundary)

                # Подаем в модель
                pred_class = model.predict(face_array)
                #Проверка работоспособности
                print(pred_class)
                self.putText(frame, f'{pred_class}', x1 + 5, y1 - 5)
        
        else:
            print('No faces detected!')
            self.putText(frame, 'No faces detected!',50, 50)

        return cv2.imshow('camera',frame)


#Реализация на практике:
rec_model = EmoRecog()
disp = VideoDisplay()


while True:
    disp.pred_emotion(rec_model)

    # Нажмите на 'q' чтобы выйти:
    if cv2.waitKey(1) & 0xff == ord('q'): 
    # Отсановим камеру и выведем сообщение об остановке
        print("[EXIT] Camera stopped")
        break
        
cv2.destroyAllWindows()