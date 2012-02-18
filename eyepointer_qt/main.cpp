#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
const char* WINDOW_NAME = "eye pointer";

int main() {
    cv::VideoCapture camera(0);
    assert(camera.isOpened());
    cv::namedWindow("eye pointer", 1);
    cv::Mat frame;
    do {
        camera >> frame;
        cv::imshow("eye pointer", frame);
    } while (cv::waitKey(10) != 27);
    return 0;
}
