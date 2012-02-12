#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    const char * winName = "Image";
    CvCapture* c = cvCreateCameraCapture(0);


    cv::Mat image = cv::imread("bleeding.jpg");
    cv::flip(image, image, 1);
    cv::namedWindow(winName);
    cv::imshow(winName, image);
    cv::waitKey(0);

    return 1;
}
