import cv2
import numpy as np

# load the video file
cap = cv2.VideoCapture('output.mp4')

# setting the background as None in the beginning
bgframe = None

while True:
    # access the frame from the video
    success,img = cap.read()

    # check if we get the frame
    if success == True:
        # resizing the frame because there is no need to process a large size frame
        img = cv2.resize(img,(600,500))
        # change the bgr frame into grayscale  format
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Applying Gaussian Blur on the Frame heavily to avoid any change in pixels intensity that
        # may occur due to camera sensor as no camera can capture a 100% accurate image.
        gray = cv2.GaussianBlur(gray,(35,35),0)

        # in the very first iteration of while loop our background (bgframe) will be None so need to provide the
        # frame as background
        if bgframe is None:
            bgframe = gray

        # Compute the difference between the first  frame (background) and current frame
        # using simple subtraction
        # delta = |background- current frame(|
        frameDelta = cv2.absdiff(bgframe, gray)

        # if any pixel value in 'frameDelta' is less than 25, we discard the pixel and set it to black (i.e. background).
        # If it is greater than 25, we’ll set it to white (i.e. foreground)
        _,thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)

        # creating a kernel of 4*4
        kernel = np.ones((7, 7), np.uint8)
        # applying errosion to avoid any small motion in video
        thresh = cv2.erode(thresh, kernel)
        # dilating our image
        thresh = cv2.dilate(thresh, None, iterations=6)
        # finding the contours
        contours, heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # finding area of contour
            area = cv2.contourArea(contour)
            print(area)
            # if area greater than the specified value the only then we will consider it
            if area > 1500:
                # find the rectangle co-ordinates
                x,y,w,h = cv2.boundingRect(contour)

                # and then dra it to indicate the moving object
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
                cv2.putText(img,'MOTION DETECTED',(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

        # displaying the frame
        cv2.imshow('VIDEO',img)
        cv2.imshow('threshold',thresh)
        cv2.waitKey(40)
    else:
        break
cap.release()
cv2.destroyAllWindows()