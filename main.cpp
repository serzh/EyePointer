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
	std::vector<int> vHist, hHist, hHistSmooth, vHistSmooth;
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
				eyeRect.height -= eyeCut * 2;		

				cv::rectangle(frame, eyeRect, CV_RGB(0, 255, 0));

				eyeImage = frame	
					.rowRange(cv::Range(eyeRect.y, eyeRect.y + eyeRect.height))
					.colRange(cv::Range(eyeRect.x, eyeRect.x + eyeRect.width));
				cv::cvtColor(eyeImage, eyeImage, CV_BGR2GRAY);

				vHist.clear();
				hHist.clear();
				vHistSmooth.clear();
				hHistSmooth.clear();

				vHist.resize(eyeImage.rows);
				hHist.resize(eyeImage.cols);
				vHistSmooth.resize(eyeImage.rows);
				hHistSmooth.resize(eyeImage.cols);

				for (int r = 0; r < eyeImage.rows; r++) {
					for (int c = 0; c < eyeImage.cols; c++) {
						vHist[r] += eyeImage.at<unsigned char>(r, c);
						hHist[c] += eyeImage.at<unsigned char>(r, c);
					}
				}

				for (int n = 0; n < 100; n ++) {
					for (int i = 1; i < hHist.size() - 1; i++) {
						hHistSmooth[i] = hHist[i-1] / 4 + hHist[i+1] / 4 + hHist[i] / 2;
					}
					for (int i = 1; i < vHist.size() - 1; i++) {
						vHistSmooth[i] = vHist[i-1] / 4 + vHist[i+1] / 4 + vHist[i] / 2;
					}
					vHist = vHistSmooth;
					hHist = hHistSmooth;
				}

				for (int i = 1; i < vHistSmooth.size() - 1; i++) {
					cv::line(frame,
							cv::Point(eyeRect.x + eyeRect.width, eyeRect.y + i),
							cv::Point(eyeRect.x + eyeRect.width + vHistSmooth[i]/200, eyeRect.y + i),
							CV_RGB(255, 0, 0));
					if (vHistSmooth[i] >= vHistSmooth[i-1] && vHistSmooth[i] >= vHistSmooth[i+1]) {
						cv::line(frame,
								cv::Point(eyeRect.x, eyeRect.y + i),
								cv::Point(eyeRect.x + eyeRect.width, eyeRect.y + i),
								CV_RGB(0, 255, 0));
					}
				}
				for (int i = 1; i < hHistSmooth.size() - 1; i++) {
					cv::line(frame,
							cv::Point(eyeRect.x + i, eyeRect.y),
							cv::Point(eyeRect.x + i, eyeRect.y - hHistSmooth[i]/50),
							CV_RGB(255, 0, 0));
					if (hHistSmooth[i] <= hHistSmooth[i-1] && hHistSmooth[i] <= hHistSmooth[i+1]) {
						cv::line(frame,
								cv::Point(eyeRect.x + i, eyeRect.y),
								cv::Point(eyeRect.x + i, eyeRect.y + eyeRect.height),
								CV_RGB(0, 255, 0));
					}
				}
				cv::imshow("eyes", eyeImage);
			}
		}
		cv::imshow( WINDOW_NAME, frame);

	} while (cv::waitKey(10) != 27);

	return 0;
}
