import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

st.title("Wig Overlay App")
st.write("Upload an image of a bald person and a wig image to see the transformation.")

# Upload images
bald_file = st.file_uploader("Upload a bald person's image", type=["jpg", "jpeg", "png"])
wig_file = st.file_uploader("Upload a wig image (with transparency)", type=["png", "jpg", "jpeg"])

def load_image(file) -> np.ndarray:
    # Open image using PIL then convert to OpenCV BGR format.
    image = Image.open(file).convert('RGB')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@st.cache(allow_output_mutation=True)
def get_face_bbox(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        # MediaPipe expects an RGB image.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        # Use the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        landmark_array = np.array(landmark_points)
        x_min, y_min = landmark_array.min(axis=0)
        x_max, y_max = landmark_array.max(axis=0)
        return (x_min, y_min, x_max, y_max)

def overlay_wig(bald_img, wig_img):
    bbox = get_face_bbox(bald_img)
    if bbox is None:
        st.error("No face detected in the image!")
        return bald_img
    x_min, y_min, x_max, y_max = bbox
    face_width = x_max - x_min

    # Set wig width 1.5x the face width and adjust height to maintain aspect ratio.
    wig_width = int(1.5 * face_width)
    wig_height = int(wig_img.shape[0] * (wig_width / wig_img.shape[1]))
    
    wig_resized = cv2.resize(wig_img, (wig_width, wig_height), interpolation=cv2.INTER_AREA)
    
    # Center wig horizontally and position it above the face.
    center_x = x_min + face_width // 2
    top_y = y_min - wig_height
    start_x = center_x - wig_width // 2

    # Convert bald image to BGRA for alpha blending.
    overlay_img = cv2.cvtColor(bald_img, cv2.COLOR_BGR2BGRA)

    end_x = start_x + wig_width
    end_y = top_y + wig_height

    # Adjust region if wig goes out of image bounds.
    if start_x < 0:
        wig_resized = wig_resized[:, abs(start_x):]
        start_x = 0
    if top_y < 0:
        wig_resized = wig_resized[abs(top_y):, :]
        top_y = 0
    if end_x > overlay_img.shape[1]:
        wig_resized = wig_resized[:, :overlay_img.shape[1]-start_x]
    if end_y > overlay_img.shape[0]:
        wig_resized = wig_resized[:overlay_img.shape[0]-top_y, :]

    # Ensure wig image has alpha channel
    if wig_resized.shape[2] == 3:
        wig_resized = cv2.cvtColor(wig_resized, cv2.COLOR_BGR2BGRA)
    
    # Extract region of interest.
    roi = overlay_img[top_y:top_y+wig_resized.shape[0], start_x:start_x+wig_resized.shape[1]]
    
    alpha_wig = wig_resized[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_wig
    
    # Blend each channel.
    for c in range(0, 3):
        roi[:, :, c] = (alpha_wig * wig_resized[:, :, c] + alpha_bg * roi[:, :, c])
    
    overlay_img[top_y:top_y+wig_resized.shape[0], start_x:start_x+wig_resized.shape[1]] = roi
    
    # Convert back to BGR format.
    result = cv2.cvtColor(overlay_img, cv2.COLOR_BGRA2BGR)
    return result

if bald_file and wig_file:
    # Load images
    bald_img = load_image(bald_file)
    # For wig, if the uploaded file is a jpg without transparency, it still works but may not blend as expected.
    wig_file_bytes = np.asarray(bytearray(wig_file.read()), dtype=np.uint8)
    wig_img = cv2.imdecode(wig_file_bytes, cv2.IMREAD_UNCHANGED)
    
    result_img = overlay_wig(bald_img, wig_img)
    
    # Convert final image to RGB for displaying with Streamlit.
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img_rgb, caption="Output with Wig", use_column_width=True)
