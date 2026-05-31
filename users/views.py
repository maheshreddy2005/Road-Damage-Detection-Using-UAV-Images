from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .forms import UserRegistrationForm
from .models import UserRegistrationModel

import os
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import load_model
#from sklearn.model_selection import train_test_split
from PIL import Image


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registration Successful")
        else:
            messages.error(request, "User already exists")
    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})



def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            request.session['userid'] = user.id
            request.session['username'] = user.name
            return render(request, 'users/UserHome.html')
        except:
            messages.error(request, "Invalid Login Details")

    return render(request, 'UserLogin.html')



def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def training(request):

    base_dir = os.path.join(settings.MEDIA_ROOT, "sih_road_dataset")

    labels = ["good", "poor", "satisfactory"]
    x, y = [], []

    for label in labels:
        folder_path = os.path.join(base_dir, label)

        # ✅ SAFETY CHECK
        if not os.path.exists(folder_path):
            return render(request, "users/training.html", {
                "error": f"Folder not found: {folder_path}"
            })

        for img in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.resize(image, (224, 224))
                x.append(image)
                y.append(labels.index(label))

            except Exception as e:
                print("Image error:", e)

    x = np.array(x) / 255.0
    y = to_categorical(np.array(y), 3)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

    model.save(os.path.join(settings.MEDIA_ROOT, "Road_Damage_model.h5"))

    return render(request, "users/training.html", {
        "message": "Model trained successfully!"
    })

def PredictRoadDamage(request):

    if request.method == 'POST':

        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/sih_road_dataset/test_data")
        filename = fs.save(image_file.name, image_file)

        image_path = os.path.join(
            settings.MEDIA_ROOT,
            'sih_road_dataset/test_data',
            filename
        )

        model = load_model(
            os.path.join(settings.MEDIA_ROOT, "Road_Damage_model.h5")
        )

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        class_labels = {
            0: "Good Road",
            1: "Poor Road",
            2: "Satisfactory Road"
        }

        result = class_labels[predicted_class]

        return render(request, "users/UploadForm.html", {
            "result": result,
            "image": "/media/sih_road_dataset/test_data/" + filename
        })

    return render(request, "users/UploadForm.html")



from django.shortcuts import render
import cv2
import numpy as np
from .utility.detector import detect_damage

def upload_image(request):
    if request.method == "POST":
        image_file = request.FILES["image"]

        np_img = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        result = detect_damage(image)

        return render(request, "users/result.html", {
            "result": result
        })

    return render(request, "users/upload.html")

from django.http import StreamingHttpResponse
from .utility.live_camera import webcam_stream

def live_camera(request):
    return StreamingHttpResponse(
        webcam_stream(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )
def live_camera(request):
    return StreamingHttpResponse(
        webcam_stream(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def live_page(request):
    return render(request, "users/live.html")
