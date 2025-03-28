import streamlit as st
import cv2
import numpy as np
import os

def overlay_transparent(background, overlay, x, y):
    """
    Overlays a transparent PNG (overlay) onto a background image at (x, y).
    Handles cases where x or y are negative by cropping the overlay appropriately.
    Both images should be in BGRA format.
    """
    bg_height, bg_width = background.shape[:2]
    ov_height, ov_width = overlay.shape[:2]

    # If x or y are negative, crop the overlay and adjust x, y
    if x < 0:
        overlay = overlay[:, -x:]
        ov_width = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        ov_height = overlay.shape[0]
        y = 0

    # If the overlay is completely outside the background, return the original background
    if x >= bg_width or y >= bg_height:
        return background

    # Calculate the region of interest dimensions
    roi_width = min(ov_width, bg_width - x)
    roi_height = min(ov_height, bg_height - y)

    # If ROI dimensions are invalid, return background
    if roi_width <= 0 or roi_height <= 0:
        return background

    overlay_crop = overlay[0:roi_height, 0:roi_width]
    overlay_img = overlay_crop[..., :3]     # BGR channels
    mask = overlay_crop[..., 3:] / 255.0      # Alpha channel normalized

    # Extract the corresponding region from the background
    background_roi = background[y:y+roi_height, x:x+roi_width, :3]
    blended = (background_roi * (1 - mask) + overlay_img * mask).astype(np.uint8)
    background[y:y+roi_height, x:x+roi_width, :3] = blended

    return background

def main():
    st.title("Put a Wig on a Bald Man!")
    st.write(
        "1. Upload an image of a bald man.\n"
        "2. The app will detect the face.\n"
        "3. The wig will be placed automatically on the head.\n"
        "4. Enjoy the new look!"
    )

    # Load the wig PNG with transparency from the local file.
    wig_path = "4.png"
    if not os.path.exists(wig_path):
        st.error("Could not find 'wig.png'. Please place your wig image in the same directory.")
        return

    wig = cv2.imread(wig_path, cv2.IMREAD_UNCHANGED)
    if wig is None or wig.shape[2] != 4:
        st.error("Error loading wig.png. Ensure it is a valid PNG with an alpha channel.")
        return

    # File uploader for the bald man image
    uploaded_file = st.file_uploader("Upload a bald man image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Error reading the image. Please try another image file.")
            return

        # Convert the image to BGRA to support alpha blending
        background = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

        # Detect face using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(background, cv2.COLOR_BGRA2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("No face detected. Try another image or adjust detection parameters.")
            st.image(cv2.cvtColor(background, cv2.COLOR_BGRA2RGB), caption="Original Image (No face found).")
        else:
            # Use the first detected face
            (x, y, w, h) = faces[0]

            # Determine the wig size (e.g., 1.2 times the face width)
            wig_width = int(1.2 * w)
            wig_height = int(wig.shape[0] * (wig_width / wig.shape[1]))
            wig_resized = cv2.resize(wig, (wig_width, wig_height), interpolation=cv2.INTER_AREA)

            # Calculate the position for the wig: shift above the face and slightly to the left
            wig_x = x - int(0.1 * w)
            wig_y = y - int(0.4 * h)

            # Overlay the wig onto the background image
            result = overlay_transparent(background.copy(), wig_resized, wig_x, wig_y)

            # Display the result in Streamlit
            st.image(cv2.cvtColor(result, cv2.COLOR_BGRA2RGB), caption="Result with Wig")

            # Provide a download option for the final image
            result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
            _, encoded_img = cv2.imencode(".png", result_bgr)
            st.download_button(
                label="Download Result",
                data=encoded_img.tobytes(),
                file_name="bald_man_with_wig.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
