#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

typedef std::vector<int> IntVec;

void smoothHist(IntVec & hist, int n);
void rectSubImg(cv::Mat & src, cv::Mat & dest, cv::Rect & rect);
void imgHist(cv::Mat & img, IntVec & hHist, IntVec & vHist);
void drawHist(cv::Mat & image, IntVec & hist, cv::Point start, char xDir, char yDir, int scale);
void splitEyeRect(cv::Rect & eyes, cv::Rect & left, cv::Rect & right);
