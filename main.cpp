#include "helpers.h"
#include <cvblob.h>

#define WINDOW_NAME "eye pointer"
#define WINDOW_RIGHT_EYE "Right eye"
#define FACE_SEARCH_WIDTH 160
#define FACE_SEARCH_HEIGHT 120

int main() {
	cv::VideoCapture camera(0);
	assert( camera.isOpened() );

	cv::CascadeClassifier faceCascade;
	assert( faceCascade.load("haarcascades/haarcascade_frontalface_alt.xml") );

	cv::CascadeClassifier eyeCascade;
	assert( eyeCascade.load("haarcascades/haarcascade_mcs_eyepair_big.xml") );

	cv::Mat frame, image, faceImage, eyeImage, rightEyeImage, leftEyeImage;
	cv::namedWindow( WINDOW_NAME, 1 );
	int threshold = 40;
	cv::createTrackbar("threshold", WINDOW_NAME, &threshold, 255);
	std::vector<cv::Rect> objects;
	IntVec vHist, hHist, rightHorizHist, rightVertHist, leftHorizHist, leftVertHist;
	cv::Rect faceRect, eyeRect, leftEye, rightEye;
	float faceModX, faceModY, eyeModX, eyeModY;
	int eyeCut;
	cvb::CvBlobs blobs;
	unsigned int result;
	IplImage iplimg;
	cvb::CvLabel largestL;
	cvb::CvBlob* largest;


	camera >> frame;
	faceModX = frame.cols / FACE_SEARCH_WIDTH;
	faceModY = frame.rows / FACE_SEARCH_HEIGHT;

	do {
		camera >> frame;

		cv::resize(frame, image, cv::Size(FACE_SEARCH_WIDTH, FACE_SEARCH_HEIGHT));
		faceCascade.detectMultiScale(image, objects);

		if ( objects.size() ) {
			faceRect = objects[0];

			faceRect.x *= faceModX;
			faceRect.y *= faceModY;
			faceRect.height *= faceModY;
			faceRect.width *= faceModX;

			cv::rectangle(frame, faceRect, CV_RGB(255, 0, 0));

			rectSubImg(frame, faceImage, faceRect);

			cv::resize(faceImage, faceImage, cv::Size(faceRect.width / 2, faceRect.height / 2));
			eyeCascade.detectMultiScale(faceImage, objects);

			if ( objects.size() ) {
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

				rectSubImg(frame, rightEyeImage, rightEye);
				rectSubImg(frame, leftEyeImage, leftEye);

				cv::cvtColor(rightEyeImage, rightEyeImage, CV_BGR2GRAY);

				cv::equalizeHist(rightEyeImage, rightEyeImage);
				cv::threshold(rightEyeImage, rightEyeImage, threshold, 255, CV_THRESH_BINARY_INV);	
				
				iplimg = rightEyeImage;
				IplImage *labels = cvCreateImage(rightEyeImage.size(), IPL_DEPTH_LABEL, 1);
				result = cvb::cvLabel(&iplimg, labels, blobs);

				largestL = cvb::cvGreaterBlob(blobs);
				largest = blobs[largestL];
				cv::rectangle(frame, cv::Point(eyeRect.x + largest->minx, eyeRect.y + largest->miny),
					cv::Point(eyeRect.x + largest->maxx, eyeRect.y + largest->maxy),
					CV_RGB(0,255,0));

				cv::imshow( WINDOW_RIGHT_EYE, rightEyeImage);
			}
		}
		cv::imshow( WINDOW_NAME, frame);

	} while (cv::waitKey(10) != 27);

	return 0;
}
