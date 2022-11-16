import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import shutil


# diabetes = pd.read_csv("diabetes.csv")
# print(diabetes.head())


## Replacing 0 with NA and replacing NA with Mean of each column
# diabetes_copy = diabetes.copy(True)
# diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
# diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace = True)
# diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace = True)
# diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].mean(), inplace = True)
# diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].mean(), inplace = True)
# diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].mean(), inplace = True)



## New Age group column and make prediction with cleaned data
# conditions = [
#     (diabetes["Age"] >= 0) & (diabetes["Age"] <= 14),
#     (diabetes["Age"] >= 15) & (diabetes["Age"] <= 24),
#     (diabetes["Age"] >= 25) & (diabetes["Age"] <= 64),
#     (diabetes["Age"] >= 64)
# ]
# values = [1,2,3,4]
# diabetes_copy["age_group"] = np.select(conditions,values)
# X = diabetes_copy[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","age_group"]]
# Y = diabetes_copy["Outcome"]
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train,Y_train)
# predict = model.predict(X_test)
# score = model.score(X_test,Y_test)
# print("\n")
# print("Actual Outcome ", ["Diabetes" if a == 1 else "No Diabetes" for a in Y_test.head(7).values])
# print("Predicted Outcome ", ["Diabetes" if p == 1 else "No Diabetes" for p in predict[:7]])
# print("This is the evaluation score: " + str(score))
# cm = confusion_matrix(Y_test,predict)
# sns.heatmap(cm,annot=True)
# plt.title("Confusion Matrix")
# plt.show()




## New Column Age Group and create bar graph based on age group and outcome
# conditions = [
#     (diabetes["Age"] >= 0) & (diabetes["Age"] <= 14),
#     (diabetes["Age"] >= 15) & (diabetes["Age"] <= 24),
#     (diabetes["Age"] >= 25) & (diabetes["Age"] <= 64),
#     (diabetes["Age"] >= 64)
# ]
# values = ["children","youth","adult","senior"]
# diabetes["age_group"] = np.select(conditions,values)
# diabetes.groupby("age_group")["Outcome"].value_counts().plot(kind="bar")
# plt.title("Diabetes Based on Age Group")
# plt.xlabel("(Age Group, Outcome)")
# plt.xticks(rotation="horizontal")
# plt.show()




## Calculating mean of different columns of the two outcome
# diabetes_copy.groupby("Outcome")["BMI"].mean().plot(kind="bar")
# plt.title("Body Mass Index and Diabetes")
# plt.xlabel("(BMI, Diabetes) ")
# print(diabetes_copy.groupby("Outcome")["Pregnancies"].mean())
# print(diabetes_copy.groupby("Outcome")["Glucose"].mean())
# print(diabetes_copy.groupby("Outcome")["BloodPressure"].mean())
# print(diabetes_copy.groupby("Outcome")["SkinThickness"].mean())
# print(diabetes_copy.groupby("Outcome")["Insulin"].mean())
# print(diabetes_copy.groupby("Outcome")["BMI"].mean())
# # diabetes_copy.groupby("Outcome")["BMI"].mean().plot(kind="bar")
# plt.show()




## Image Detection Training with diabetic retinopathy
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# img_height = 32
# img_width = 32
# batch_size = 32
# class_names = ["Diabetes", "No Diabetes"]
# 
# model = keras.Sequential(
#     [
#         layers.Input((32, 32, 3)),
#         layers.Conv2D(32, 3, padding="same"),
#         layers.Conv2D(64, 3, padding="same"),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(2),
#     ]
# )
#
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     "data/",
#     labels="inferred",
#     label_mode="int",  # categorical, binary
#     class_names=['diabetes','no_diabetes'],
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="training",
# )
#
# ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
#     "data/",
#     labels="inferred",
#     label_mode="int",  # categorical, binary
#     color_mode="rgb",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="validation",
# )
#
#
# def augment(x, y):
#     image = tf.image.random_brightness(x, max_delta=0.05)
#     return image, y
#
#
# ds_train = ds_train.map(augment)
#
# # # Custom Loops
# # for epochs in range(10):
# #     for x, y in ds_train:
# #         pass
#
#
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
#     metrics=["accuracy"],
# )
#
#
# model.fit(ds_train, epochs=10, verbose=2)
# test_loss, test_acc = model.evaluate(np.concatenate([img for img, lbl in ds_validation], axis=0),np.concatenate([lbl for img, lbl in ds_validation], axis=0),verbose=2)
# print(test_acc)
#
#
# IMG_INDEX = -1
# predictions = model.predict(ds_validation)
# for images, labels in list(ds_validation):
#     for img , lbl in zip(images, labels):
#         IMG_INDEX += 1
#         plt.title("Predicted Result: " + class_names[np.argmax(predictions[IMG_INDEX])])
#         plt.xlabel("Actual Result: " + class_names[lbl.numpy()])
#         plt.imshow(img.numpy().astype("uint8"))
#         plt.grid(False)
#         plt.show()




## Image Division Script
# Source of trainLabels.csv: https://www.kaggle.com/c/diabetic-retinopathy-detection/data

# directory = "data/images/"
# destination_0 = "data/non_diabetes/"
# destination_1 = "data/diabetes/"
# data = pd.read_csv("trainLabels.csv")
# list_images = os.listdir(directory)
# for list_image in list_images:
#     level = data.loc[data["image"] == list_image.split(".")[0], "level"].iloc[0]
#     if level == 0:
#         shutil.move(directory + list_image, destination_0 + list_image )
#     else:
#         shutil.move(directory + list_image, destination_1 + list_image)

