#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>

#define WINDOW_NAME "eye pointer"
#define FACE_SEARCH_WIDTH 160
#define FACE_SEARCH_HEIGHT 120

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

typedef std::vector<int> IntVec;

void smoothHist(IntVec & hist, int n) {
	if (n <= 0)
		return;

	IntVec smooth;
	smooth.resize(hist.size());
	while (n) {
		for (int i = 1; i < hist.size() - 1; i++) {
			smooth[i] = hist[i-1] / 4 + hist[i+1] / 4 + hist[i] / 2;
		}
		hist = smooth;
		n--;
	}
}

cv::Mat rectSubImg(cv::Mat src, cv::Rect rect) {
	return src
		.rowRange(cv::Range(rect.y, rect.y + rect.height))
		.colRange(cv::Range(rect.x, rect.x + rect.width));
}

void imgHist(cv::Mat img, IntVec & hHist, IntVec & vHist) {
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, CV_BGR2GRAY);

	vHist.clear();
	hHist.clear();

	vHist.resize(grayImg.rows);
	hHist.resize(grayImg.cols);

	for (int r = 0; r < grayImg.rows; r++) {
		for (int c = 0; c < grayImg.cols; c++) {
			vHist[r] += grayImg.at<unsigned char>(r, c);
			hHist[c] += grayImg.at<unsigned char>(r, c);
		}
	}
}

void drawHist(cv::Mat & image, IntVec & hist, cv::Point start, char xDir, char yDir, int scale = 1) {
	cv::Point end;
	for (int i = 0; i < hist.size(); i ++) {
		end = start;
		switch(yDir) {
			case UP:
				end.y -= hist[i] / scale;
				break;
			case DOWN:
				end.y += hist[i] / scale;
				break;
			case LEFT:
				end.x -= hist[i] / scale;
				break;
			case RIGHT:
				end.x += hist[i] / scale;
				break;
		}
		cv::line(image, start, end, CV_RGB(255, 255, 255));
		switch(xDir) {
			case UP:
				start.y --;
				break;
			case DOWN:
				start.y ++;
				break;
			case LEFT:
				start.x --;
				break;
			case RIGHT:
				start.x ++;
				break;
		}
	}
}

void splitEyeRect(cv::Rect eyes, cv::Rect & left, cv::Rect & right) {
	int vTrim = eyes.height / 5, hTrim = eyes.width / 10;
	eyes.y += vTrim;
	eyes.height -= vTrim * 2;

	left = eyes;
	left.width /= 2;

	right = left;
	right.x += left.width;

	right.x += hTrim;

	left.width -= hTrim;
	right.width -= hTrim;

}

int main()
{
	cv::VideoCapture camera(0);
	assert( camera.isOpened() );

	cv::CascadeClassifier faceCascade;
	assert( faceCascade.load("haarcascades/haarcascade_frontalface_alt.xml") );

	cv::CascadeClassifier eyeCascade;
	assert( eyeCascade.load("haarcascades/haarcascade_mcs_eyepair_big.xml") );

	cv::Mat frame, image, faceImage, eyeImage, rightEyeImage, leftEyeImage;
	cv::namedWindow( WINDOW_NAME, 1 );
	std::vector<cv::Rect> objects;
	IntVec vHist, hHist, rightHorizHist, rightVertHist, leftHorizHist, leftVertHist;
	cv::Rect faceRect, eyeRect, leftEye, rightEye;
	float faceModX, faceModY, eyeModX, eyeModY;
	int eyeCut;

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

			faceImage = rectSubImg(frame, faceRect);

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

				splitEyeRect(eyeRect, leftEye, rightEye);

				cv::rectangle(frame, leftEye, CV_RGB(255, 255, 0));
				cv::rectangle(frame, rightEye, CV_RGB(255, 255, 0));

				rightEyeImage = rectSubImg(frame, rightEye);
				leftEyeImage = rectSubImg(frame, leftEye);

				imgHist(rightEyeImage, rightHorizHist, rightVertHist);
				imgHist(leftEyeImage, leftHorizHist, leftVertHist);

				smoothHist(rightHorizHist, 100);
				smoothHist(rightVertHist, 100);
				smoothHist(leftHorizHist, 100);
				smoothHist(leftVertHist, 100);

				drawHist(frame, rightHorizHist, cv::Point(rightEye.x, rightEye.y), RIGHT, UP, 100);
				drawHist(frame, rightVertHist, cv::Point(rightEye.x + rightEye.width, rightEye.y), DOWN, RIGHT, 100);

				drawHist(frame, leftHorizHist, cv::Point(leftEye.x, leftEye.y), RIGHT, UP, 100);
				drawHist(frame, leftVertHist, cv::Point(leftEye.x, leftEye.y), DOWN, LEFT, 100);
			}
		}
		cv::imshow( WINDOW_NAME, frame);

	} while (cv::waitKey(10) != 27);

	return 0;
}
