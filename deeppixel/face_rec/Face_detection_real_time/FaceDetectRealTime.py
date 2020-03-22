import cv2

def realTimeFaceRecognition():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Converts to gray 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Quits when 'Q' or 'q' or 'esc' is pressed 
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    realTimeFaceRecognition()


if __name__ == '__main__': 
    main()