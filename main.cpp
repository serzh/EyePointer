#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>

#define WINDOW_NAME "eye pointer"
#define FACE_CASCADE_FILE "haarcascades/haarcascade_frontalface_alt.xml"
#define EYE_CASCADE_FILE "haarcascades/haarcascade_mcs_eyepair_big.xml"

int main() {
    cv::VideoCapture camera(0);
    assert( camera.isOpened() );

    cv::CascadeClassifier faceCascade;
    assert( faceCascade.load(FACE_CASCADE_FILE) );

    cv::CascadeClassifier eyeCascade;
    assert( eyeCascade.load(EYE_CASCADE_FILE) );

    cv::Mat frame, image, faceImage;
    cv::namedWindow( WINDOW_NAME, 1 );

    std::vector<cv::Rect> objects;
    cv::Rect faceRect, eyeRect;

    do {
        //get frame from camera
        camera >> frame;

        // resize image to decrese processing time
        cv::resize(frame, image, cv::Size(160, 120));

        // detect face
        faceCascade.detectMultiScale(image, objects);

        if (objects.size()) { // if face found
            faceRect = objects[0];

            // scale face rectangle to frame size
            faceRect.x *= 4;
            faceRect.y *= 4;
            faceRect.height *=4;
            faceRect.width *= 4;

            // draw face rectangle
            cv::rectangle(frame, faceRect, CV_RGB(255, 0, 0));

            // get face subimage
            faceImage = frame
                    .rowRange(cv::Range(faceRect.y, faceRect.y + faceRect.height))
                    .colRange(cv::Range(faceRect.x, faceRect.x + faceRect.width));

            // resize image to decrease processing time
            cv::resize(faceImage, image, cv::Size(faceRect.width / 2, faceRect.height / 2));

            // detect eyes
            eyeCascade.detectMultiScale(image, objects);

            if (objects.size()) { // if eyes founs
                eyeRect = objects[0];

                // scale eyes rectangle to frame size
                eyeRect.x *= 2;
                eyeRect.y *= 2;
                eyeRect.height *= 2;
                eyeRect.width *= 2;

                // offset eyes rectangle to draw it on main frame
                eyeRect.x += faceRect.x;
                eyeRect.y += faceRect.y;

                //draw eyes rectangle
                cv::rectangle(frame, eyeRect, CV_RGB(0, 255, 0));
            }
        }

        //show frame
        cv::imshow( WINDOW_NAME, frame);

    } while (cv::waitKey(10) != 27); // 27 is ESC key

    return 0;
}
