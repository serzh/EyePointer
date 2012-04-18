#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <iostream>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define MAIN_WINDOW "Eyer"
#define TEMPLATE_WINDOW "Template"
#define NEW_SIZE_WIDTH 320
#define NEW_SIZE_HEIGHT 240

void crop(cv::Mat & src, cv::Mat & dst, cv::Rect roi) {
	dst = src.
		rowRange(roi.y, roi.y + roi.height).
		colRange(roi.x, roi.x + roi.width);
}
void movePointer(int xp, int yp) {
    static Display *display = XOpenDisplay(NULL);
    static Window root = DefaultRootWindow(display);

	int x, y, tmp;
	unsigned int tmpmask;
	Window fromroot, tmpwin;

	XQueryPointer(display, root, &fromroot, &tmpwin, &x, &y, &tmp, &tmp, &tmpmask);

	x += xp;
	y += yp;

	XWarpPointer(display, None, root, 0, 0, 0, 0, x, y);
	XFlush(display);
}

int main(int argc, char **argv) {

	int counter = 0;
	cv::Rect eyePairRect;
	cv::Mat frame, miniFrame, eyePair, result;
	cv::CascadeClassifier eyesCascade("haarcascades/haarcascade_mcs_eyepair_big.xml");
	std::vector<cv::Rect> eyePairs;
	double minval, maxval;
	int dx, dy;
	cv::Point minloc, maxloc, current, previous;

	cv::VideoCapture camera(0);
	assert( camera.isOpened() );

	cv::namedWindow(MAIN_WINDOW, 1);
	cv::namedWindow(TEMPLATE_WINDOW, 1);

	camera >> frame;

	float scalex = frame.size().width / NEW_SIZE_WIDTH;
	float scaley = frame.size().height / NEW_SIZE_HEIGHT;

	do {

		camera >> frame;

		cv::resize(frame, miniFrame, cv::Size(NEW_SIZE_WIDTH, NEW_SIZE_HEIGHT));
		cv::cvtColor(miniFrame, miniFrame, CV_RGB2GRAY);
		cv::equalizeHist(miniFrame, miniFrame);

		if (!(counter % 50)) {

			eyesCascade.detectMultiScale(miniFrame, eyePairs);

			if (eyePairs.size()) 
				eyePairRect = eyePairs[0];
			else continue;

		} else {

			cv::matchTemplate(miniFrame, eyePair, result, CV_TM_SQDIFF);

			cv::minMaxLoc(result, &minval, &maxval, &minloc, &maxloc, cv::Mat());

			eyePairRect.x = minloc.x;
			eyePairRect.y = minloc.y;
			eyePairRect.width = eyePair.size().width;
			eyePairRect.height = eyePair.size().height;

		}

		crop(miniFrame, eyePair, eyePairRect);

		eyePairRect.x *= scalex;
		eyePairRect.y *= scaley;
		eyePairRect.width *= scalex;
		eyePairRect.height *= scaley;
		
		cv::rectangle(frame, cv::Point(eyePairRect.x, eyePairRect.y),
		 		cv::Point(eyePairRect.x + eyePairRect.width, eyePairRect.y + eyePairRect.height),
				CV_RGB(255, 0, 255));

		current = cv::Point(eyePairRect.x + eyePairRect.width/2, eyePairRect.y + eyePairRect.height/2);

		if (!(counter % 50)) {
			previous = current;
		}

		cv::circle(frame, current, 3, CV_RGB(128, 128, 255));

		dx = current.x - previous.x;
		dy = current.y - previous.y;

		movePointer(-dx*9, dy*8);

		cv::imshow(MAIN_WINDOW, frame);
		cv::imshow(TEMPLATE_WINDOW, eyePair);

		previous = current;

		counter++;

	} while (cv::waitKey(1) != 27);

}