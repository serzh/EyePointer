/*
 * real.cpp
 *
 *  Created on: Feb 2, 2012
 *      Author: serzh
 */
#include <cv.h>
#include <highgui.h>

const char* WINDOW_NAME = "Find eye";

int main(int argc, char **argv) {

	cvNamedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);
	CvCapture* c = cvCreateCameraCapture(0);

	IplImage* prev = cvQueryFrame(c);
	IplImage* curr;
	CvSize size = cvGetSize(prev);
	IplImage* prev_gray = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* curr_gray = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* diff = cvCreateImage(size, IPL_DEPTH_8U, 1);

	//	This first variant
/*	while (true) {

		prev = cvQueryFrame(c);
		cvCvtColor(prev, prev_gray, CV_RGB2GRAY);

		cvWaitKey(1);

		curr = cvQueryFrame(c);
		cvCvtColor(curr, curr_gray, CV_RGB2GRAY);

		cvAbsDiff(curr_gray, prev_gray, diff);
		cvThreshold(diff, diff, 20, 255, CV_THRESH_BINARY);
		IplConvKernel* kernel;
		kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
		cvMorphologyEx(diff, diff, NULL, kernel, CV_MOP_OPEN, 1);

		cvShowImage(WINDOW_NAME, diff);

	}
	int p_pos = 0;
	*p1 = 100;
	cvCreateTrackbar("param1", WINDOW_NAME, &p_pos, 200, onTackbarSlide);
*/

	// This is second variant (maybe better)
	while (true) {

		prev = cvQueryFrame(c);
		cvCvtColor(prev, prev_gray, CV_RGB2GRAY);

		cvWaitKey(1);

		curr = cvQueryFrame(c);
		cvCvtColor(curr, curr_gray, CV_RGB2GRAY);

		cvAbsDiff(curr_gray, prev_gray, diff);
		cvSmooth(diff, diff, CV_GAUSSIAN, 9, 9);

		cvCanny(diff, diff, 10, 10, 7);

		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq* circles = cvHoughCircles(curr_gray, storage, CV_HOUGH_GRADIENT, 1, 70, 70, 20, 20,30);

		int i;
		for (i = 0; i < circles->total; i++) {
			float* p = (float*) cvGetSeqElem(circles, i);
			cvCircle(curr_gray, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 3, 8, 0);
		}

		cvShowImage(WINDOW_NAME, diff);
		cvWaitKey(1);
	}
	// End of second variant

	cvReleaseCapture(&c);
	cvDestroyWindow("Find eye");
}

