#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>

#define WINDOW_NAME "eye pointer"
#define FACE_SEARCH_WIDTH 160
#define FACE_SEARCH_HEIGHT 120

void smoothHist(std::vector<int> & hist, int n) {
	if (n <= 0)
		return;

	std::vector<int> smooth;
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

void imgHist(cv::Mat img, std::vector<int> & hHist, std::vector<int> & vHist) {
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

	cv::Mat frame, image, faceImage, eyeImage;
	cv::namedWindow( WINDOW_NAME, 1 );
	std::vector<cv::Rect> objects;
	std::vector<int> vHist, hHist;
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

				eyeImage = rectSubImg(frame, eyeRect);
				imgHist(eyeImage, hHist, vHist);
				smoothHist(vHist, 100);
				smoothHist(hHist, 100);

				//for (int i = 1; i < vHist.size() - 1; i++) {
				//	cv::line(frame,
				//			cv::Point(eyeRect.x + eyeRect.width, eyeRect.y + i),
				//			cv::Point(eyeRect.x + eyeRect.width + vHist[i]/200, eyeRect.y + i),
				//			CV_RGB(255, 0, 0));
				//	if (vHist[i] >= vHist[i-1] && vHist[i] >= vHist[i+1]) {
				//		cv::line(frame,
				//				cv::Point(eyeRect.x, eyeRect.y + i),
				//				cv::Point(eyeRect.x + eyeRect.width, eyeRect.y + i),
				//				CV_RGB(0, 255, 0));
				//	}
				//}
				//for (int i = 1; i < hHist.size() - 1; i++) {
				//	cv::line(frame,
				//			cv::Point(eyeRect.x + i, eyeRect.y),
				//			cv::Point(eyeRect.x + i, eyeRect.y - hHist[i]/50),
				//			CV_RGB(255, 0, 0));
				//	if (hHist[i] <= hHist[i-1] && hHist[i] <= hHist[i+1]) {
				//		cv::line(frame,
				//				cv::Point(eyeRect.x + i, eyeRect.y),
				//				cv::Point(eyeRect.x + i, eyeRect.y + eyeRect.height),
				//				CV_RGB(0, 255, 0));
				//	}
				//}
			}
		}
		cv::imshow( WINDOW_NAME, frame);

	} while (cv::waitKey(10) != 27);

	return 0;
}
