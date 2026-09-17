import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# LOAD YOLO SEGMENTATION MODEL
# =========================================================

print("Loading YOLO segmentation model...")

model = YOLO("yolo26n-seg.pt")

print("Model loaded!")


# =========================================================
# OPEN MACBOOK CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("Camera started.")
print("Hold an object in front of the camera.")
print("Press Q to quit.")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # =====================================================
    # YOLO SEGMENTATION
    # =====================================================

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.45,
        verbose=False
    )


    # Copy original frame
    output = frame.copy()


    # =====================================================
    # PROCESS DETECTED OBJECTS
    # =====================================================

    if results[0].masks is not None:

        masks = results[0].masks.data
        boxes = results[0].boxes

        for i, mask in enumerate(masks):

            # ---------------------------------------------
            # GET CLASS INFORMATION
            # ---------------------------------------------

            class_id = int(boxes.cls[i])

            class_name = model.names[class_id]

            confidence = float(boxes.conf[i])


            # ---------------------------------------------
            # RESIZE MASK TO CAMERA FRAME
            # ---------------------------------------------

            mask = mask.cpu().numpy()

            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0])
            )


            # ---------------------------------------------
            # CONVERT TO BINARY MASK
            # ---------------------------------------------

            mask = (mask > 0.5).astype(np.uint8) * 255


            # ---------------------------------------------
            # SMOOTH EDGES
            # ---------------------------------------------

            kernel = np.ones((5, 5), np.uint8)

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            mask = cv2.GaussianBlur(
                mask,
                (11, 11),
                0
            )


            # =================================================
            # CREATE INVISIBILITY EFFECT
            # =================================================

            # OpenCV inpainting attempts to reconstruct
            # the pixels covered by the detected object.

            invisible = cv2.inpaint(
                frame,
                mask,
                15,
                cv2.INPAINT_TELEA
            )


            # -------------------------------------------------
            # BLEND THE ORIGINAL FRAME AND INPAINTED FRAME
            # -------------------------------------------------

            alpha = mask.astype(np.float32) / 255.0

            alpha = cv2.merge([
                alpha,
                alpha,
                alpha
            ])


            output = (
                frame.astype(np.float32) * (1 - alpha)
                +
                invisible.astype(np.float32) * alpha
            )

            output = np.uint8(output)


            # =================================================
            # DISPLAY DETECTION INFORMATION
            # =================================================

            cv2.putText(
                output,
                f"{class_name} {confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    cv2.imshow(
        "YOLO Invisible Object",
        output
    )


    # =====================================================
    # QUIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")