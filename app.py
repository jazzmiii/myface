import base64
import os

app_data_path = '/app/app_data'

# Make sure the necessary directories exist and have proper permissions
os.makedirs(f'{app_data_path}/ultralytics', exist_ok=True)
os.makedirs(f'{app_data_path}/.keras', exist_ok=True)

# Set environment variables for the respective paths
os.environ['ULTRALYTICS_CONFIG_DIR'] = f'{app_data_path}/ultralytics'
os.environ['KERAS_HOME'] = f'{app_data_path}/.keras'

import cv2
import numpy as np
from flask import Flask, request, jsonify
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from flask_pymongo import PyMongo
from ultralytics import YOLO
from bson.objectid import ObjectId  # Import ObjectId for MongoDB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize MTCNN detector and FaceNet model
detector = MTCNN()
embedder = FaceNet()

# Configure MongoDB
app.config["MONGO_URI"] = "mongodb+srv://nanduvinay719:76qqKRX4zC97yQun@travis.744fuyn.mongodb.net/?retryWrites=true&w=majority&appName=travis" #config setting
# Update with your MongoDB URI
mongo = PyMongo(app)#initialize
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_cnn_embedding_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='linear')  # Final dense layer to produce a 512-dimensional embedding
    ])
    return model
cnn_model = create_cnn_embedding_model()
cnn_model.compile(optimizer='adam', loss='mse') 
# Using MSE loss as we are not training for classification
def cosine(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity
@app.route('/CNN-login', methods=['POST'])
def cnnlogin():
    # Check for uploaded image
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    rollnumber = request.form['rollnumber']
    rollnumber = rollnumber.upper()

    if not rollnumber:
        return jsonify({"error": "Roll number is required"}), 400

    # Fetch user data from MongoDB
    user_data = mongo.db.data.find_one({"RollNumber": rollnumber}, {"CNN_embeddings": 1, "username": 1})
    if user_data is None:
        return jsonify({"error": "User not found"}), 404
    stored_cnn_embedding = np.array(user_data["CNN_embeddings"])# Convert stored embeddings to NumPy array
    username = user_data["username"]
    print(username)
    # Decode image
    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    # Initialize Haar Cascade

    # Detect the face using Haar Cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    # Process the first detected face
    x, y, w, h = faces[0]
    cropped_face = cv2.resize(image[y:y+h, x:x+w], (160, 160))

    # Preprocess face for CNN model
    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY).reshape(160, 160, 1)
    normalized_face = np.expand_dims(gray_face, axis=-1)
    normalized_face = np.expand_dims(normalized_face, axis=0)
    normalized_face = normalized_face / 255.0  # Normalize pixel values to [0, 1]
    # Generate embedding using the CNN model
    cnn_embedding = cnn_model.predict(normalized_face)[0]
    print("CNN shape: ",cnn_embedding.shape,"Stored shape: ",stored_cnn_embedding.shape)
    # Compare the embedding with stored embeddings
    similarity = cosine(cnn_embedding, stored_cnn_embedding)
    print(f"Similarity: {similarity}")

    # Set a threshold for recognition
    recognition_threshold = 0.94  # Adjust this threshold as needed
    if similarity > recognition_threshold:
        return jsonify({"name": username, "probability": float(similarity)}), 200
    else:
        return jsonify({"error": "Face not recognized", "probability": float(similarity)}), 401

@app.route('/login', methods=['POST'])
def recognizeLogin():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    name = request.form["rollnumber"]
    name = name.upper()
    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)   
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    results = recognize_faces_in_image(image)
    print(results[0])
    print(results[0]['name'],name)
    if (results[0]['name']==name):
        print("done")
        today = datetime.now().strftime("%Y-%m-%d")
        mongo.db.attendance1.update_one(
                {'username': results[0]['name']},
                {'$set': {today: True}},  # Mark as present for today
                upsert=True
            )
        return jsonify({"name": name, "probability": results[0]['probability']}), 200
    else:
        return jsonify({'name':"user not recognised"})

@app.route('/register', methods=['POST'])
def register():
    rollnumber = request.form["RollNumber"]
    username = request.form['Username']
    fathername = request.form["FatherName"]
    phoneno = request.form["phoneNumber"]
    facenet_embeddings, cnn_embeddings = [], []
    stored_image = None  # To store the first grayscale image

    print(username)
    # Check if user already exists
    existing_user = mongo.db.data.find_one({"username": username}, {"_id": 0})
    if existing_user:
        return jsonify({"error": f"User '{username}' already exists"}), 400

    # Process uploaded images
    for i in range(5):  # Expecting 5 images
        try:
            image_file = request.files[f'image{i}']
        except KeyError:
            return jsonify({"error": f"Missing image{i} in the request"}), 400

        image_data = image_file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Face detection using MTCNN for FaceNet
        mtcnn_faces = detector.detect_faces(image)
        if mtcnn_faces:
            # Get the first detected face for FaceNet embedding
            x, y, w, h = mtcnn_faces[0]['box']
            x, y = max(0, x), max(0, y)
            cropped_face = cv2.resize(image[y:y+h, x:x+w], (160, 160))
            rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

            # Get FaceNet embedding
            facenet_embedding = embedder.embeddings(np.expand_dims(rgb_face, axis=0)).flatten()
            facenet_embeddings.append(facenet_embedding)
        # Face detection using Haar Cascade for CNN
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        if len(haar_faces) > 0:
            x, y, w, h = haar_faces[0]
            cropped_face = cv2.resize(image[y:y+h, x:x+w], (160, 160))
            gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY).reshape(160, 160, 1)
            normalized_face=np.expand_dims(gray_face, axis=-1) #dimnesion
            normalized_face=np.expand_dims(normalized_face, axis=0) #batch
            normalized_face = normalized_face / 255.0 #0-1
            # Get CNN embedding
            cnn_embedding = cnn_model.predict(normalized_face)[0]
            cnn_embeddings.append(cnn_embedding)
            # Save the first grayscale face as base64
            if stored_image is None:
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY))
                stored_image = base64.b64encode(buffer).decode('utf-8')

    if not facenet_embeddings or not cnn_embeddings:
        return jsonify({"error": "No valid faces detected in the uploaded images"}), 400

    # Calculate mean embeddings
    mean_facenet_embedding = np.mean(facenet_embeddings, axis=0).astype(float).tolist()
    mean_cnn_embedding = np.mean(cnn_embeddings, axis=0).astype(float).tolist()
    cnn_model.save_weights('cnn_model.weights.h5') 
    # Create user data
    id = mongo.db.data.count_documents({}) + 1
    user_data = {
        'RollNumber': rollnumber,
        'username': username,
        'FatherName': fathername,
        'phoneNumber': phoneno,
        'embeddings': mean_facenet_embedding,
        'CNN_embeddings': mean_cnn_embedding,
        'stored_image': stored_image,
        'id': id
    }

    # Insert into MongoDB
    mongo.db.data.insert_one(user_data)
    mongo.db.attendance1.insert_one({"username": rollnumber, "id": id})
    # Reload embeddings
    reload_embeddings()

    return jsonify({"message": "User registered successfully!"}), 201

# Load embeddings from MongoDB for recognition
@app.route('/get_users', methods=['GET'])
def get_users():
    users = list(mongo.db.data.find({}, {"id": 1, "username": 1}))  # Fetch only _id and username
    user_count = len(users)
    return jsonify({"users": [user['username'] for user in users], "count": user_count})

def load_embeddings_from_db():
    users = list(mongo.db.data.find())
    face_data = []# facenet embeddings
    labels = [] # id 1,2,3,..
    names = {} #dict of id and roll number
    # {"1":vinay,"2":shahank}
    for user in users:
        face_data.append(user["embeddings"])
        labels.append(user['id'])  # Keep the ObjectId
        names[user['id']] = user['RollNumber']  # Use ObjectId as key

    return (face_data, labels, names) if face_data else ([], [], {})

# Load face embeddings from MongoDB initially
face_data, labels, names = load_embeddings_from_db()
cnn_model.load_weights('cnn_model.weights.h5')



# Reload embeddings to update after a new registration
def reload_embeddings():
    global face_data, labels, names
    cnn_model.load_weights('cnn_model.weights.h5')
    face_data, labels, names = load_embeddings_from_db()

# Recognize faces using MongoDB-stored embeddings
model = YOLO('yolov5s.pt')  # Replace with your YOLO model path


@app.route('/crowd', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read the image from the request
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Perform YOLO detection
    results = model.predict(source=img, conf=0.5) 
    print(results)# Confidence threshold
    detections = results[0].boxes.xyxy  # Bounding boxes
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    human_boxes = [box for box, label in zip(detections, labels) if int(label) == 0]  # Filter humans

    # Draw bounding boxes for humans only
    human_count = 0
    for box in human_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        human_count += 1

    # Convert the processed image to Base64
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'count': human_count, 'image': encoded_image})

def recognize_faces_in_image(image):
    if len(face_data) == 0:
        return [{"name": "No registered faces", "probability": 0.0}]

    faces = detector.detect_faces(image)
    results = []
    for face in faces:
        x, y, width, height = face['box']
        cropped_face = cv2.resize(image[y:y+height, x:x+width], (160, 160))
        
        # Convert cropped face to RGB
        rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        embedding = embedder.embeddings(np.expand_dims(rgb_face, axis=0)).flatten()  # Use RGB face here

        # Compare with stored embeddings in MongoDB
        similarities = cosine_similarity([embedding], face_data)
        idx = np.argmax(similarities)
        best_match = similarities[0][idx]

        if best_match > 0.7:
            recognized_id = labels[idx] # Get the ObjectId
            recognized_name = names[recognized_id]  # Use ObjectId to get the username
            results.append({"name": recognized_name, "probability": float(best_match)})
        else:
            results.append({"name": "Unknown", "probability": float(best_match)})
    return results

@app.route('/users/<username>/images', methods=['GET'])
def get_user_images(username):
    username = str(username).upper()
    details = mongo.db.data.find_one({"username": username},{"_id":0,"embeddings":0,"CNN_embeddings":0})
    if not details:
        details = mongo.db.data.find_one({"RollNumber": username},{"_id":0,"embeddings":0,"CNN_embeddings":0})
        if not details:
          return jsonify({"error": "User not found"}), 404
    print(details["RollNumber"])
    # Retrieve the stored image in base64 format
    return jsonify({"details": details})

#multi face
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    results = recognize_faces_in_image(image)
    today = datetime.now().strftime("%Y-%m-%d")
    for result in results:
        if result['name'] != "Unknown":  # Only log attendance for recognized users
            mongo.db.attendance1.update_one(
                {'username': result['name']},
                {'$set': {today: True}},  # Mark as present for today
                upsert=True
            )
    return jsonify(results)

# user multi face
@app.route('/user_recognize', methods=['POST'])
def user_recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"},), 400

    file = request.files['image']
    image_data = file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    results = recognize_faces_in_image(image)
    return jsonify(results)

@app.route('/user_attendance/<username>', methods=['GET'])
def get_user_attendance(username):
    # Check if the user exists in the database
    user = mongo.db.data.find_one({'RollNumber': username})
    if user is None:
        return jsonify({"error": "User not found"}), 404
    print(user['username'])
    # Fetch the attendance data for the user
    attendance = mongo.db.attendance1.find_one({'username': username}, {'_id': 0,"username":0,"id":0})
    print(attendance)
    if attendance is None:
        return jsonify({"error": "No attendance data found"}), 404

    # Return the attendance data
    return jsonify(attendance)

#attendance
@app.route('/attendance',methods=['GET'])
def get_attendance():
    records = list(mongo.db.attendance1.find({}, {"_id": 0}))
    return jsonify({"attendance": records})

