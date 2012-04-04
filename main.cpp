#include "helpers.h"
#define WINDOW_NAME "eye pointer"
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
	std::vector<cv::Rect> objects;
	IntVec vHist, hHist, rightHorizHist, rightVertHist, leftHorizHist, leftVertHist;
	cv::Rect faceRect, eyeRect, leftEye, rightEye;
	float faceModX, faceModY, eyeModX, eyeModY;
	int eyeCut;

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


				//FIXME move to functions
				for(int i = 1; i < rightHorizHist.size() - 1; i ++) {
					if (rightHorizHist[i] <= rightHorizHist[i-1] && rightHorizHist[i] <= rightHorizHist[i+1]) {
						cv::line(frame,
								cv::Point(rightEye.x + i, rightEye.y),
								cv::Point(rightEye.x + i, rightEye.y + rightEye.height),
								CV_RGB(255, 0, 0));
					}
				}

				for(int i = 1; i < leftHorizHist.size() - 1; i ++) {
					if (leftHorizHist[i] <= leftHorizHist[i-1] && leftHorizHist[i] <= leftHorizHist[i+1]) {
						cv::line(frame,
								cv::Point(leftEye.x + i, leftEye.y),
								cv::Point(leftEye.x + i, leftEye.y + leftEye.height),
								CV_RGB(255, 0, 0));
					}
				}

				for(int i = 1; i < rightVertHist.size() - 1; i ++) {
					if (rightVertHist[i] >= rightVertHist[i-1] && rightVertHist[i] >= rightVertHist[i+1]) {
						cv::line(frame,
								cv::Point(rightEye.x, rightEye.y + i),
								cv::Point(rightEye.x + rightEye.width, rightEye.y + i),
								CV_RGB(255, 0, 0));
					}
				}

				for(int i = 1; i < leftVertHist.size() - 1; i ++) {
					if (leftVertHist[i] >= leftVertHist[i-1] && leftVertHist[i] >= leftVertHist[i+1]) {
						cv::line(frame,
								cv::Point(leftEye.x, leftEye.y + i),
								cv::Point(leftEye.x + leftEye.width, leftEye.y + i),
								CV_RGB(255, 0, 0));
					}
				}
			}
		}
		cv::imshow( WINDOW_NAME, frame);

	} while (cv::waitKey(10) != 27);

	return 0;
}
