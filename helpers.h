#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cvblob.h>

#include <vector>
#include <iostream>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

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
void findLargestBlob(cv::Mat & src, cv::Rect & largest);
int mean(int arr[], int size);
void zeros(int arr[], int size);
void movePointer(int xp, int yp);
