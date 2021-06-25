from tensorflow.keras.applications import VGG19, VGG16

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.activations import swish
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 사전 학습된 모델 불러오기
model = VGG16(weights='imagenet', include_top=False, input_shape = (256,256,3))
for layer in model.layers[:-2]:
    layer.trainable = False

x = model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512)(x)
x = swish(x)
x = Dropout(0.4)(x)
x = Dense(256)(x)
x = swish(x)
x = Dropout(0.4)(x)
x = Dense(4, activation='softmax')(x)

# new model 정의
new_model = Model(inputs = model.input, outputs = x)
new_model.compile(loss='categorical_crossentropy',
                     optimizer=Adam(learning_rate=0.0001),
                     metrics=['accuracy'])

train_image_generator = ImageDataGenerator(
            rescale=1./255, horizontal_flip=True, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, fill_mode='nearest')
test_image_generator = ImageDataGenerator(rescale=1./255)

train_dir ='./OCT2017 /train'
test_dir = './OCT2017 /test'
val_dir = './OCT2017 /val'

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(256, 256))


val_data_gen = test_image_generator.flow_from_directory(directory=val_dir,
                                                       target_size=(256, 256))


test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                         target_size=(256, 256))

# 모델 학습
history = new_model.fit(train_data_gen, 
                        epochs=7, 
                        validation_data=val_data_gen)               

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

from matplotlib import pyplot as plt

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig("Train History")

scores = new_model.evaluate_generator(test_data_gen)

print("Loss of the model: %.2f"%(scores[0]))
print("Test Accuracy: %.2f%%"%(scores[1] * 100))


new_model.save('./new_model.h5')
