import cv2
import numpy as np
import time


class FireDetector():
    def __init__(self):
        cv2.namedWindow("window1")
        self.capture = cv2.VideoCapture(0)
        #self.cam_fps = self.capture.get(cv2.CAP_PROP_FPS)
        #print('This camera has: {} FPS.'.format(self.cam_fps))
    def run(self):
        avg_fps_counter = 0
        seconds = 0
        fps = 0
        while True:
            start = time.time()
            frame_ok, frame = self.capture.read()
            if not frame_ok:
                continue

            pframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            pframe = cv2.GaussianBlur(pframe, (5,5), 0)

            red_lower = np.array([40, 120, 80], np.uint8)
            red_upper = np.array([60, 255, 255], np.uint8)

            res,pframe = cv2.threshold(pframe, 180, 255, cv2.THRESH_BINARY)

            dilation = np.ones((5, 5), "uint8")
            pframe = cv2.dilate(pframe, dilation)

            image, contours, hierarchy = cv2.findContours(pframe, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            max_area = 0
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = contour

            if largest_contour is not None:
                #print('Found light!')

                moment = cv2.moments(largest_contour)
                #print(moment['m00'])

                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

                cv2.circle(frame, (int(rect[0][0]), int(rect[0][1])), 10, (0, 255, 0), -1)

                height, width = frame.shape[:2]
                cv2.line(frame, (int(rect[0][0]), int(rect[0][1])), (int(rect[0][0]), height), (0,0,255), 2)
                cv2.line(frame, (int(rect[0][0]), int(rect[0][1])), (int(rect[0][0]), 0), (0, 0, 255), 2)

                cv2.line(frame, (int(rect[0][0]), int(rect[0][1])), (width, int(rect[0][1])), (0,0,255), 2)
                cv2.line(frame, (int(rect[0][0]), int(rect[0][1])), (0, int(rect[0][1])), (0, 0, 255), 2)

            else:
                #print('Cant see anything...')
                pass


            end = time.time()
            seconds += (end - start)

            if avg_fps_counter == 5:
                fps = 5 / seconds
                seconds = 0
                avg_fps_counter = 0
            else:
                avg_fps_counter += 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, '{:.3f}'.format(fps), (0, 25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("window1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.capture.release()

if __name__ == '__main__':
    fireDetector = FireDetector()
    fireDetector.run()
