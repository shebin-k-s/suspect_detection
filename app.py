import os
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, request, jsonify
import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import shutil
import logging

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MARKED_IMAGES = 'marked_images'
DB_PATH = 'database'
TEMP_FOLDER = 'temp_faces'
FRAME_SKIP = 30
MODEL_NAME = 'VGG-Face'
CONFIDENCE_THRESHOLD = 0.25
DISTANCE_THRESHOLD = 0.80
MAX_FACES_PER_FRAME = 20

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MARKED_IMAGES, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load DNN face detector
dnn_face_detector_proto = "models/deploy.prototxt"
dnn_face_detector_model = "models/res10_300x300_ssd_iter_140000.caffemodel"

try:
    if os.path.exists(dnn_face_detector_model) and os.path.exists(dnn_face_detector_proto):
        face_detector = cv2.dnn.readNetFromCaffe(dnn_face_detector_proto, dnn_face_detector_model)
        use_dnn = True
        logger.info("Using DNN face detector (more accurate)")
    else:
        logger.warning("DNN model files not found, falling back to Haar Cascade")
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        use_dnn = False
except Exception as e:
    logger.error(f"Error loading DNN model: {e}, falling back to Haar Cascade")
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    use_dnn = False

def normalize_name(name):
    """Normalize suspect names to lowercase for consistent handling."""
    if name:
        return name.lower().strip()
    return None

def find_case_insensitive_path(base_dir, name):
    """Find a file or directory in base_dir regardless of case."""
    if not name:
        return None
    
    name_lower = normalize_name(name)
    for item in os.listdir(base_dir):
        if normalize_name(item) == name_lower:
            return os.path.join(base_dir, item)
    return None

def clean_database():
    """Ensure all images in the database are valid JPEG images and contain faces."""
    logger.info("Cleaning and validating database...")
    invalid_count = 0
    
    for file in os.listdir(DB_PATH):
        img_path = os.path.join(DB_PATH, file)
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            
            if use_dnn:
                faces = detect_face_dnn(img_array)
                has_face = faces and len(faces) > 0
            else:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                has_face = len(faces) > 0
            
            if has_face:
                # Ensure filename is lowercase to maintain consistency
                lowercase_filename = normalize_name(os.path.basename(img_path))
                lowercase_path = os.path.join(DB_PATH, lowercase_filename)
                
                # Save with lowercase filename
                img.save(lowercase_path, format="JPEG")
                
                # Remove original if case is different
                if lowercase_path != img_path:
                    os.remove(img_path)
                    
                logger.info(f"Validated image: {lowercase_filename}")
            else:
                logger.warning(f"No face detected in database image: {file}")
                os.remove(img_path)
                invalid_count += 1
                
        except Exception as e:
            logger.error(f"Invalid image removed: {file} ({e})")
            os.remove(img_path)
            invalid_count += 1
    
    logger.info(f"Database cleaned. Removed {invalid_count} invalid images.")
    
    if len(os.listdir(DB_PATH)) == 0:
        logger.warning("WARNING: No valid face images found in database. Please add suspect images.")

def detect_face_dnn(frame):
    """Detect faces using DNN model for better accuracy."""
    if not use_dnn:
        return None
        
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    faces = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.65:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            if startX < w and startY < h and endX > 0 and endY > 0:
                if (endX - startX) >= 70 and (endY - startY) >= 70:
                    faces.append((startX, startY, endX - startX, endY - startY, confidence))
    
    return faces

def get_face_embeddings(face_img_path):
    """Get face embedding vector using DeepFace."""
    try:
        embedding_objs = DeepFace.represent(face_img_path, model_name=MODEL_NAME, enforce_detection=False)
        if embedding_objs:
            return embedding_objs[0]["embedding"]
        return None
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def verify_match(face_path, suspect_path, threshold=DISTANCE_THRESHOLD):
    """Verify if two faces match with custom threshold."""
    try:
        result = DeepFace.verify(face_path, suspect_path, model_name=MODEL_NAME, 
                                enforce_detection=False, distance_metric="cosine")
        
        distance = result.get("distance", 1.0)
        verified = result.get("verified", False)
        
        if verified and distance < threshold:
            confidence = 1 - distance
            logger.info(f"Match confidence: {confidence:.2f} (distance: {distance:.4f})")
            return True, confidence
        return False, 0
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False, 0

def should_rotate_frame(frame):
    """Determine whether the frame needs rotation based on aspect ratio and face detection."""
    height, width = frame.shape[:2]

    # Check if the frame is in portrait mode (height > width)
    if height < width:
        return True
    
    return False

def mark_suspects(video_path, device_id):
    """Detects faces in a video and checks against the suspect database."""
    # Normalize device_id to lowercase for consistency
    device_id = normalize_name(device_id)
    
    cap = cv2.VideoCapture(video_path)
    detected_suspects = {}
    suspect_images = {}
    frame_count = 0
    processed_frames = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if len(os.listdir(DB_PATH)) == 0:
        logger.error("ERROR: No suspect images in database. Please add suspects first.")
        return {
            "error": "No suspect images in database",
            "message": "Please add suspect images before processing videos"
        }
    
    suspect_paths = [os.path.join(DB_PATH, f) for f in os.listdir(DB_PATH)]
    # Extract normalized suspect names from filenames
    suspect_names = [normalize_name(os.path.splitext(os.path.basename(p))[0]) for p in suspect_paths]
    
    logger.info(f"Processing video with {total_frames} frames at {fps} FPS (analyzing every {FRAME_SKIP}th frame)")
    logger.info(f"Database contains {len(suspect_names)} suspects: {', '.join(suspect_names)}")

    rotate = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
       
        if rotate is None:
            rotate = should_rotate_frame(frame)
            logger.info(f"Auto-detected rotation: {'Yes' if rotate else 'No'}")

        # Apply rotation if required
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        processed_frames += 1
        
        if processed_frames % 10 == 0:
            logger.info(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        
        marked_frame = frame.copy()
        frame_suspects = {}

        faces = []
        if use_dnn:
            faces = detect_face_dnn(frame)
            if faces is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_haar = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                faces = [(x, y, w, h, 1.0) for (x, y, w, h) in faces_haar]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_haar = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            faces = [(x, y, w, h, 1.0) for (x, y, w, h) in faces_haar]
            
        if faces and len(faces) > MAX_FACES_PER_FRAME:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:MAX_FACES_PER_FRAME]

        for face_idx, face_info in enumerate(faces):
            if len(face_info) == 5:
                x, y, w, h, confidence = face_info
            else:
                x, y, w, h = face_info[:4]
                confidence = 1.0
            
            if confidence < 0.7:
                continue
                
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0 or face_crop.shape[0] < 60 or face_crop.shape[1] < 60:
                continue
                
            face_id = f"{frame_count}_{x}_{y}_{face_idx}"
            temp_face_path = os.path.join(TEMP_FOLDER, f"temp_face_{face_id}.jpg")
            cv2.imwrite(temp_face_path, face_crop)

            try:
                best_match = None
                best_confidence = 0
                
                for suspect_idx, suspect_path in enumerate(suspect_paths):
                    suspect_name = suspect_names[suspect_idx]
                    
                    is_match, confidence = verify_match(temp_face_path, suspect_path)
                    
                    if is_match and confidence > best_confidence:
                        best_match = suspect_name
                        best_confidence = confidence

                if best_match and best_confidence >= CONFIDENCE_THRESHOLD:
                    suspect_name = normalize_name(best_match)
                    logger.info(f"SUSPECT DETECTED: {suspect_name} (confidence: {best_confidence:.2f})")

                    if suspect_name not in detected_suspects:
                        detected_suspects[suspect_name] = True
                        suspect_images[suspect_name] = []
                    
                    frame_suspects[suspect_name] = best_confidence

                    cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    conf_text = f"{suspect_name} ({best_confidence:.2f})"
                    cv2.putText(marked_frame, conf_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            except Exception as e:
                logger.error(f"Face matching error on face {face_id}: {e}")
                continue

            finally:
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                    
        if frame_suspects:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = f"{timestamp}_{device_id}_frame{frame_count}.jpg"
            
            for suspect in frame_suspects.keys():
                normalized_suspect = normalize_name(suspect)
                # Create suspect folder with normalized name
                suspect_folder = os.path.join(MARKED_IMAGES, normalized_suspect)
                os.makedirs(suspect_folder, exist_ok=True)
                
                suspect_image_path = os.path.join(suspect_folder, frame_filename)
                cv2.imwrite(suspect_image_path, marked_frame)
                suspect_images[normalized_suspect].append(suspect_image_path)

    cap.release()

    all_image_paths = []
    for suspect, paths in suspect_images.items():
        all_image_paths.extend(paths)

    return {
        "message": "Video processing complete",
        "suspects_detected": list(detected_suspects.keys()),
        "total_frames_processed": processed_frames,
        "marked_frames": len(all_image_paths),
        "suspect_images": suspect_images
    }

@app.route('/add_suspect', methods=['POST'])
def add_suspect():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Missing image or name"}), 400

    # Convert name to lowercase for consistent storage
    name = normalize_name(request.form['name'])
    image = request.files['image']

    try:
        img = Image.open(image).convert("RGB")
        img_array = np.array(img)
        
        if use_dnn:
            faces = detect_face_dnn(img_array)
            has_face = faces is not None and len(faces) > 0
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            has_face = len(faces) > 0
        
        if not has_face:
            return jsonify({
                "error": "No clear face detected in the image", 
                "details": "Please upload an image with a clear frontal face"
            }), 400
            
        if use_dnn and faces is not None and len(faces) > 1:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face[:4]
        elif not use_dnn and len(faces) > 1:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
        else:
            if use_dnn and faces is not None:
                x, y, w, h = faces[0][:4]
            else:
                x, y, w, h = faces[0]
                
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_array.shape[1], x + w + pad_w)
        y2 = min(img_array.shape[0], y + h + pad_h)
        
        face_img = img_array[y1:y2, x1:x2]
        face_img_pil = Image.fromarray(face_img)
        
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    # Use normalized name for filename
    filename = secure_filename(name + ".jpg")
    save_path = os.path.join(DB_PATH, filename)
    face_img_pil.save(save_path, format="JPEG", quality=95)

    return jsonify({
        "message": "Suspect added successfully", 
        "suspect_name": name, 
        "image_path": save_path,
        "face_detected": True
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """API endpoint to upload a video and process it for suspect detection."""
    if 'video' not in request.files or 'device' not in request.form:
        return jsonify({"error": "No file or device info uploaded"}), 400

    file = request.files['video']
    # Normalize device ID to lowercase
    device_id = normalize_name(request.form['device'])

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(video_path)

    try:
        result = mark_suspects(video_path, device_id)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    return jsonify(result)

    
@app.route('/fetch_suspects', methods=['GET'])
def query_suspects():
    suspect_name = request.args.get('suspect_name', None)
    if suspect_name:
        suspect_name = normalize_name(suspect_name)
        
    device_id = request.args.get('device', None)
    if device_id:
        device_id = normalize_name(device_id)
        
    time_start = request.args.get('time_start', None)
    time_end = request.args.get('time_end', None)
    
    # Get all suspect folders in MARKED_IMAGES directory
    suspect_folders = [f for f in os.listdir(MARKED_IMAGES) if os.path.isdir(os.path.join(MARKED_IMAGES, f))]
    
    results = {}
    
    # If specific suspect is requested, find it case-insensitively
    if suspect_name:
        # Look for the folder with case-insensitive matching
        suspect_folder_path = find_case_insensitive_path(MARKED_IMAGES, suspect_name)
        if suspect_folder_path:
            suspect_folders = [os.path.basename(suspect_folder_path)]
        else:
            # No matching folder found
            return jsonify({
                "message": "No matching suspect found",
                "results": {}
            })
    
    for suspect in suspect_folders:
        normalized_suspect = normalize_name(suspect)
        suspect_folder = os.path.join(MARKED_IMAGES, suspect)
        results[normalized_suspect] = {"devices": {}, "total_sightings": 0, "images": []}
        
        for image_file in os.listdir(suspect_folder):
            # Image filename format: timestamp_deviceID_frameX.jpg
            # e.g. 20250314_123045_camera1_frame150.jpg
            try:
                parts = image_file.split('_')
                if len(parts) >= 3:
                    img_date = parts[0]
                    img_time = parts[1]
                    img_device = normalize_name(parts[2])
                    img_frame = parts[3].replace('frame', '').replace('.jpg', '')
                    
                    timestamp = f"{img_date}_{img_time}"
                    
                    # Apply device filter if specified (case-insensitive)
                    if device_id and img_device != device_id:
                        continue
                    
                    # Apply time filter if specified
                    if time_start and timestamp < time_start:
                        continue
                    if time_end and timestamp > time_end:
                        continue
                    
                    # Add to device count
                    if img_device not in results[normalized_suspect]["devices"]:
                        results[normalized_suspect]["devices"][img_device] = 0
                    
                    results[normalized_suspect]["devices"][img_device] += 1
                    results[normalized_suspect]["total_sightings"] += 1
                    
                    # Add image path
                    image_path = os.path.join(suspect_folder, image_file)
                    results[normalized_suspect]["images"].append({
                        "path": image_path,
                        "timestamp": timestamp,
                        "device": img_device,
                        "frame": img_frame
                    })
            except Exception as e:
                logger.error(f"Error parsing image filename {image_file}: {e}")
    
    # Filter out suspects with no matching results
    results = {k: v for k, v in results.items() if v["total_sightings"] > 0}
    
    return jsonify({
        "message": "Suspect query completed",
        "results": results
    })

if __name__ == '__main__':
    clean_database()
    logger.info(f"Starting surveillance app with {'DNN' if use_dnn else 'Haar Cascade'} face detector")
    logger.info(f"Settings: confidence_threshold={CONFIDENCE_THRESHOLD}, distance_threshold={DISTANCE_THRESHOLD}")
    app.run(debug=True)