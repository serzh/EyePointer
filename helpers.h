#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cvblob.h>

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
void findLargestBlob(cv::Mat & src, cv::Rect & largest);
int mean(int arr[], int size);
void zeros(int arr[], int size);

typedef std::pair<cv::Point, cv::Point> Pair;

class PupilsFinder {

private:
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;
	int threshold;
	cv::Mat frame, image, faceImage, eyeImage, rightEyeImage, leftEyeImage;
	std::vector<cv::Rect> objects;
	cv::Rect faceRect, eyeRect, leftEye, rightEye;
	float faceModX, faceModY;
	cvb::CvBlobs blobs;
	IplImage iplimg;
	cvb::CvLabel largestL;
	cvb::CvBlob* largest;
	cv::Rect rightPupilRect, leftPupilRect;
	int resizeW, resizeH;
	cv::Point r, l;
	char key;
	int rx, ry, lx, ly;
	int counter, frames;
	Pair pupils;

public:
	PupilsFinder();
	void setFaceCascade(cv::CascadeClassifier & faceCascade);
	void setEyeCascade(cv::CascadeClassifier & eyeCascade);
	void setThreshold(int threshold);
	void setResize(int resizeW, int resizeH);
	void setFaceMods(float faceModX, float faceModY);
	Pair find(cv::Mat & frame, bool drawBorder);
	Pair loop(cv::VideoCapture & camera, std::string wname, int iterations, int aprxIterations, bool drawBorder);
	void loopf(cv::VideoCapture & camera, std::string wname, void (*f)(int, int, int, int), int iterations, int aprxIterations, bool drawBorder);
};

class Mover {

private:
	int kw, kh;
	std::string wname;
	int screen_w, screen_h;
	cv::Mat bg;
	int crx, cry;
public:
	Mover(int kw, int kh);
	void fun(int rx, int ry, int lx, int ly);
};
