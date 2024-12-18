import tensorflow
import numpy as np
import os
from django.conf import settings
from django.shortcuts import render
from PIL import Image

# Class names in English
class_names = [
    'Horse', 'Cow', 'Chicken', 'Squirrel', 'Sheep', 'Butterfly',
    'Spider', 'Dog', 'Elephant', 'Cat'
]

def predict_image(img_path):
    try:
        model_path = os.path.join(settings.BASE_DIR, 'image_prediction/multiclass.keras')  # Adjust the path if needed
        print("Before model load")
        model = tensorflow.keras.models.load_model(model_path)
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load and preprocess the image
    img = tensorflow.keras.preprocessing.image.load_img(img_path, target_size=(100, 100))  # Resize to match model's expected input
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = tensorflow.keras.applications.resnet50.preprocess_input(img_array)  # Preprocess the image
    print("Before prediction")

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class index with the highest probability
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    print("After prediction")

    # Get the predicted class name in English
    predicted_class = class_names[predicted_class_idx]

    return predicted_class

def upload_image(request):
    if request.method == "POST" and request.FILES['image']:
        uploaded_file = request.FILES['image']

        try:
            # Save the uploaded file temporarily
            temp_file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(temp_file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            print("Image uploaded")

            # Predict the class of the uploaded image
            prediction = predict_image(temp_file_path)
            if prediction:
                print("Prediction:", prediction)
                # Return the result page with the prediction
                return render(request, 'result.html', {'prediction': prediction})
            else:
                return render(request, 'upload.html', {'error': 'Model loading failed.'})

        except Exception as e:
            print("Error:", e)
            return render(request, 'upload.html', {'error': 'An error occurred during prediction.'})

    return render(request, 'upload.html')
