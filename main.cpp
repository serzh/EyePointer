#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>

#define WINDOW_NAME "eye pointer"
#define FACE_SEARCH_WIDTH 160
#define FACE_SEARCH_HEIGHT 120

int main()
{
    cv::VideoCapture camera(0);
    assert( camera.isOpened() );

    cv::CascadeClassifier faceCascade;
    assert( faceCascade.load("haarcascades/haarcascade_frontalface_alt.xml") );

    cv::CascadeClassifier eyeCascade;
    assert( eyeCascade.load("haarcascades/haarcascade_mcs_eyepair_big.xml") );

    cv::Mat frame, image, faceImage;
    cv::namedWindow( WINDOW_NAME, 1 );
    std::vector<cv::Rect> objects;
    cv::Rect faceRect, eyeRect;
    float faceModX, faceModY, eyeModX, eyeModY;

    camera >> frame;
    faceModX = frame.cols / FACE_SEARCH_WIDTH;
    faceModY = frame.rows / FACE_SEARCH_HEIGHT;

    do
    {
        camera >> frame;

        cv::resize(frame, image, cv::Size(FACE_SEARCH_WIDTH, FACE_SEARCH_HEIGHT));
        faceCascade.detectMultiScale(image, objects);

        if ( objects.size() )
	{
            faceRect = objects[0];

            faceRect.x *= faceModX;
            faceRect.y *= faceModY;
            faceRect.height *= faceModY;
            faceRect.width *= faceModX;

            cv::rectangle(frame, faceRect, CV_RGB(255, 0, 0));

            faceImage = frame
                    .rowRange(cv::Range(faceRect.y, faceRect.y + faceRect.height))
                    .colRange(cv::Range(faceRect.x, faceRect.x + faceRect.width));

            cv::resize(faceImage, faceImage, cv::Size(faceRect.width / 2, faceRect.height / 2));
            eyeCascade.detectMultiScale(faceImage, objects);

            if ( objects.size() )
	    {
                eyeRect = objects[0];

                eyeRect.x *= 2;
                eyeRect.y *= 2;
                eyeRect.height *= 2;
                eyeRect.width *= 2;

                eyeRect.x += faceRect.x;
                eyeRect.y += faceRect.y;

                cv::rectangle(frame, eyeRect, CV_RGB(0, 255, 0));
            }
        }

        cv::imshow( WINDOW_NAME, frame);

    } while (cv::waitKey(10) != 27); // 27 is ESC key

    return 0;
}
