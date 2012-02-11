#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {

    // read an image
    cv::Mat image= cv::imread("img.jpg");
    // create image window named "My Image"
    cv::namedWindow("My Image");
    // show the image on window
    cv::imshow("My Image", image);
    // wait key for 5000 ms
    cv::waitKey(5000);

    return 1;
}
