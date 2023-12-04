import streamlit as st
import cv2
import os
from PIL import Image
import joblib
import tempfile

model = joblib.load('svm_model.pkl')

pca = joblib.load('pca_model.joblib')

scaler = joblib.load('scaler.pkl')


def preprocess_and_image(face_classifier: cv2.CascadeClassifier, image, target_size=(64, 64), x=100, y=100, crop_width=800, crop_height=800):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(105, 105))

    if len(faces) > 0:

        x, y, w, h = faces[0]


        img = img[y:y + h, x:x + w]

    else:
        print(f"No face detected in {image}. Performing a normal crop.")
        img = img[y:y + crop_height, x:x + crop_width]


    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    

    return img

def predict(image_path):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    img = preprocess_and_image(face_classifier, image_path)
    preprocessed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the preprocessed image using Streamlit's st.image()
    st.image(preprocessed_image, caption='Preprocessed Image', use_column_width=True)

    
    flattened_image = preprocessed_image.flatten() 
    Normalize = flattened_image.astype('float32') / 255.0

    scaled_image = scaler.transform([Normalize])

    pca_transformed = pca.transform(scaled_image)

    prediction = model.predict(pca_transformed)

    return prediction
def main():
    st.title('Your Streamlit App Title')
    st.sidebar.title('Sidebar Title')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, 'uploaded_image.jpg')

        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())

        image = Image.open(temp_path)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        prediction = predict(temp_path)

        temp_dir.cleanup()

        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
