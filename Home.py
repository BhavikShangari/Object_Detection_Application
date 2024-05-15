import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Reshape, Dropout, Flatten
B = 2
N_CLASSES = 20
H, W = 448, 448
SPLIT_SIZE = H // 32
NUM_FILTERS = 512
OUTPUT_DIM = N_CLASSES + 5 * B

classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
         'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

@st.cache_resource
def load_model():

    base_model = tf.keras.applications.efficientnet.EfficientNetB1(
        weights='imagenet',
        input_shape=(H, W, 3),
        include_top=False,
    )
    base_model.trainable = False

    model = tf.keras.Sequential([    
        base_model,
        Conv2D(NUM_FILTERS, (3, 3), padding='same', kernel_initializer='he_normal',),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        Conv2D(NUM_FILTERS, (3, 3), padding='same', kernel_initializer='he_normal',),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        Conv2D(NUM_FILTERS, (3, 3), padding='same', kernel_initializer='he_normal',),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        Conv2D(NUM_FILTERS, (3, 3), padding='same', kernel_initializer='he_normal',),
        LeakyReLU(alpha=0.1),

        Flatten(),
        
        Dense(NUM_FILTERS, kernel_initializer='he_normal',),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        Dropout(0.5),
        
        Dense(SPLIT_SIZE * SPLIT_SIZE * OUTPUT_DIM, activation='sigmoid'),
        
        Reshape((SPLIT_SIZE, SPLIT_SIZE, OUTPUT_DIM)),
    ])
    
    model.load_weights('./model/yolo_efficientnet_b1_new.h5')
    print(model.summary())
    return model


model = load_model()


def model_test(img):
    try:
    
        output = model.predict(np.expand_dims(img, axis=0))


        THRESH = .25

        object_positions = tf.concat(
            [tf.where(output[..., 0] >= THRESH), tf.where(output[..., 5] >= THRESH)], axis=0)
        # print(object_positions)
        selected_output = tf.gather_nd(output, object_positions)
        # print(selected_output)
        final_boxes = []
        final_scores = []

        for i, pos in enumerate(object_positions):
            for j in range(2): 
                if selected_output[i][j * 5] > THRESH:
                    output_box = tf.cast(output[pos[0]][pos[1]][pos[2]][(j * 5) + 1:(j * 5) + 5], dtype=tf.float32)
                    
                    x_centre = (tf.cast(pos[1], dtype=tf.float32) + output_box[0]) * 32
                    y_centre = (tf.cast(pos[2], dtype=tf.float32) + output_box[1]) * 32

                    x_width, y_height = tf.math.abs(H * output_box[2]), tf.math.abs(W * output_box[3])
                    
                    x_min, y_min = int(x_centre - (x_width / 2)), int(y_centre - (y_height / 2))
                    x_max, y_max = int(x_centre + (x_width / 2)), int(y_centre + (y_height / 2))

                    if(x_min <= 0):x_min = 0
                    if(y_min <= 0):y_min = 0
                    if(x_max >= W):x_max = W
                    if(y_max >= H):y_max = H
                    final_boxes.append(
                        [x_min, y_min, x_max, y_max,
                        str(classes[tf.argmax(selected_output[..., 10:], axis=-1)[i]])])
                    final_scores.append(selected_output[i][j * 5])

        final_boxes = np.array(final_boxes)
        
        object_classes = final_boxes[..., 4]
        nms_boxes = final_boxes[..., 0:4]

        nms_output = tf.image.non_max_suppression(
            nms_boxes, final_scores, max_output_size=100, iou_threshold=0.2,
            score_threshold=float('-inf')
        )
        img = img.numpy()
        # print(nms_output)
        for i in nms_output:
            cv2.rectangle(
                img,
                (int(final_boxes[i][0]), int(final_boxes[i][1])),
                (int(final_boxes[i][2]), int(final_boxes[i][3])), (0, 0, 255), 1)
            cv2.putText(
                img,
                final_boxes[i][-1],
                (int(final_boxes[i][0]), int(final_boxes[i][1]) + 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 200, 2), 1
                )
        return img
    except Exception as e:
        st.text("NO object found !!!")


st.title('Object Detection Framework')
size = None
frame = st.file_uploader('Give your Image', type=['jpeg', 'jpg', 'png'])
if frame is not None:
    img = np.array(Image.open(frame))
    size = img.shape[:2]
    img = tf.image.resize(tf.constant(img), [448, 448])

    st.header('Original Image')
    st.image(frame, use_column_width='always')
    
    st.header('Processed Image')
    final_img = model_test(img)
    if final_img is not None:
        st.image(Image.fromarray(np.uint8(tf.image.resize(final_img, size).numpy())), use_column_width='always')
