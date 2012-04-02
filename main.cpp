#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>

#define WINDOW_NAME "eye pointer"
#define FACE_SEARCH_WIDTH 160
#define FACE_SEARCH_HEIGHT 120

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
	cv::namedWindow( "eyes", 1 );
	std::vector<cv::Rect> objects;
	std::vector<int> vHist, hHist;
	cv::Rect faceRect, eyeRect;
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

			faceImage = frame
				.rowRange(cv::Range(faceRect.y, faceRect.y + faceRect.height))
				.colRange(cv::Range(faceRect.x, faceRect.x + faceRect.width));

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

				eyeCut = eyeRect.height / 5;
				eyeRect.y += eyeCut;
				eyeRect.height -= eyeCut;		

				cv::rectangle(frame, eyeRect, CV_RGB(0, 255, 0));

				eyeImage = frame	
					.rowRange(cv::Range(eyeRect.y, eyeRect.y + eyeRect.height))
					.colRange(cv::Range(eyeRect.x, eyeRect.x + eyeRect.width));
				cv::cvtColor(eyeImage, eyeImage, CV_BGR2GRAY);
				vHist.clear();
				hHist.clear();
				vHist.resize(eyeImage.rows);
				hHist.resize(eyeImage.cols);
				for (int r = 0; r < eyeImage.rows; r++) {
					for (int c = 0; c < eyeImage.cols; c++) {
						vHist[r] += eyeImage.at<unsigned char>(r, c);
						hHist[c] += eyeImage.at<unsigned char>(r, c);
					}
				}
				for (int i = 0; i < vHist.size(); i++) {
					cv::line(frame,
							cv::Point(eyeRect.x + eyeRect.width, eyeRect.y + i),
							cv::Point(eyeRect.x + eyeRect.width + vHist[i]/200, eyeRect.y + i),
							CV_RGB(255, 0, 0));
				}
				for (int i = 0; i < hHist.size(); i++) {
					cv::line(frame,
							cv::Point(eyeRect.x + i, eyeRect.y),
							cv::Point(eyeRect.x + i, eyeRect.y - hHist[i]/100),
							CV_RGB(255, 0, 0));
				}
				cv::imshow("eyes", eyeImage);
			}
		}
		cv::imshow( WINDOW_NAME, frame);

	} while (cv::waitKey(10) != 27);

	return 0;
}
