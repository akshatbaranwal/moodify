# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# %%
model = tf.keras.models.load_model("model/model.h5")

with open("model/history", "rb") as file:
    h = pickle.load(file)
with open("model/parameters", "rb") as file:
    p = pickle.load(file)
with open("model/epochs", "rb") as file:
    e = pickle.load(file)


class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


history = History_trained_model(h, e, p)


# %%
mapper = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}


# %%
# Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face2 = gray

    for (x, y, w, h) in faces:
        face2 = gray[y : y + h, x : x + w]
        face2 = cv2.resize(face2, (48, 48))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img,
            mapper[np.argmax(model.predict(face2.reshape(1, 48, 48, 1))[0], axis=-1)],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (36, 255, 12),
            2,
        )
        print(model.predict(face2.reshape(1, 48, 48, 1))[0])

    cv2.imshow("img", img)

    # Stop if escape key is pressed
    # k = cv2.waitKey(30) & 0xFF
    # if k == 27:
    #     break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture object
cap.release()

# %%
