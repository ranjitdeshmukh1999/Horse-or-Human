# Horse-or-Human

import tensorflow as tf

train_dr="E:\\My Datasets\\Image Datasets\\horse-or-human\\train\\"
test_dr="E:\\My Datasets\\Image Datasets\\horse-or-human\\validation\\"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gene=ImageDataGenerator(rescale=1./255.)
test_gener=ImageDataGenerator(rescale=1./255.)

train_generator=train_gene.flow_from_directory(train_dr,batch_size=20,class_mode='binary',target_size=(150,150))
test_generator=test_gener.flow_from_directory(test_dr,batch_size=20,class_mode='binary',target_size=(150,150))

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(72, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(), 
    
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dropout(0.1, seed=2019),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dropout(0.2, seed=2019),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dropout(0.3, seed=2019),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.4, seed=2019),
    
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.summary()


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
                        verbose=1, mode='auto',restore_best_weights=True)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['acc'])


history = model.fit(train_generator,
                              validation_data=test_generator,
                              steps_per_epoch=20,epochs=10,
                              validation_steps=10,
                              verbose=1,callbacks=[monitor])


test2_datagen  = ImageDataGenerator( rescale = 1.0/255. )
test_dir = "E:\\My Datasets\\Image Datasets\\horse-or-human\\my\\"
test_generator =  test2_datagen.flow_from_directory(test_dir,
                                                    batch_size=6,
                                                    class_mode  = None,
                                                    target_size = (150, 150),
                                                    shuffle=False)

y_prob = model.predict(test_generator,callbacks=[monitor])
y_pred = ["HUMAN" if probs > 0.5 else "HORSE" for probs in y_prob]
y_pred


y_prob

y_prob

y_pred



