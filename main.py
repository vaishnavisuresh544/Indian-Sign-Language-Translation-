import cv2
import time

x0 = 400
y0 = 200
height = 200
width = 200
isBgModeOn = 0
isAdaptiveThresholdMode = True
roi = None
isPredictionMode = True
model= None
menu = "\n c-Change Filter\n p-Predict Sign\n w-Move ROI Upside\n s-Move ROI Downside\n a-Move ROI Rightside\n d-Move ROI Leftside\n ESC-exit\n"
def adaptiveThresholdMode(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]

    if roi is None:
        print("Error: Failed to extract ROI from the frame.")
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    res = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return res

# Function for SIFT mode
def siftMode(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]

    if roi is None:
        print("Error: Failed to extract ROI from the frame.")
        return None

    return roi

# Function for no filter mode
def noFilterMode(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]

    if roi is None:
        print("Error: Failed to extract ROI from the frame.")
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    res = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    sift = cv2.SIFT_create()
    kp = sift.detect(res, None)
    res = cv2.drawKeypoints(res, kp, res, (0, 0, 255))

    return res
def Main():
    global isBgModeOn, x0, y0, roi, isPredictionMode, model, menu

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Please make sure your webcam is connected.")
        return

    ret = cap.set(3, 640)
    ret = cap.set(4, 480)
    if not ret:
        print("Error: Failed to set camera parameters.")

    print(menu)

    model = create_cnn_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read a frame from the camera.")
            break

        frame = cv2.flip(frame, 3)

        if isBgModeOn == 0:
            roi = adaptiveThresholdMode(frame, x0, y0, width, height)
        elif isBgModeOn == 1:
            roi = siftMode(frame, x0, y0, width, height)
        else:
            roi = noFilterMode(frame, x0, y0, width, height)

        if isPredictionMode:
            if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                result = predictSign(roi, model)
                cv2.putText(frame, result, (10, 355 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 1)
                time.sleep(0.04)

                # Display predicted sign in a separate window
                cv2.imshow('Predicted Sign', roi)
            else:
                print("Error: Invalid ROI for prediction")

        cv2.imshow('Sign Language Detector', frame)

        key = cv2.waitKey(10) & 0xff

        if key == ord('c'):
            if isBgModeOn == 2:
                isBgModeOn = 0
            else:
                isBgModeOn = isBgModeOn + 1
            if isBgModeOn == 0:
                print("Adaptive Threshold Mode active")
            elif isBgModeOn == 1:
                print("Sift Mode active")
            else:
                print("No Filter Mode active")
            if isPredictionMode:
                model = create_cnn_model()
        elif key == ord('p'):
            isPredictionMode = not isPredictionMode
            if isPredictionMode:
                model = create_cnn_model()
            print("Prediction Mode - {}".format(isPredictionMode))
        elif key == ord('w'):
            y0 = y0 - 5
        elif key == ord('s'):
            y0 = y0 + 5
        elif key == ord('a'):
            x0 = x0 - 5
        elif key == ord('d'):
            x0 = x0 + 5
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
