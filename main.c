/*
 * real.cpp
 *
 *  Created on: Feb 2, 2012
 *      Author: serzh
 */
#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

 const char* WINDOW_NAME = "Find eye";
 const char* WINDOW_TMP = "Template";
 int x = 20;

 IplImage* crop(IplImage* src, CvRect rect) {
 	IplImage* cropped = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
 	cvSetImageROI(src, rect);
 	cvCopy(src, cropped);
 	cvResetImageROI(src);

 	return cropped;
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

 	int s = 0;
 	int i;
 	CvRect *rt_l;
 	CvSeq* seq_l;
 	IplImage* cropped_l;
 	IplImage* res;
 	CvSize sizeR;
 	IplImage* curr_gray_b;
 	IplImage* curr_gray;


 	CvPoint minloc, maxloc;
 	double minval, maxval;

 	cvNamedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);
 	cvNamedWindow(WINDOW_TMP, CV_WINDOW_AUTOSIZE);
 	cvCreateTrackbar("Eye size",WINDOW_NAME, &x, 20);
 	CvCapture* c = cvCreateCameraCapture(0);
 	char* filename_l = "haarcascades/haarcascade_mcs_eyepair_big.xml";
 	char* filename_r = "haarcascades/haarcascade_righteye_2splits.xml";

 	// Make all good things for left eye
 	CvHaarClassifierCascade* cascade_l = 
 	(CvHaarClassifierCascade*) cvLoad(filename_l, 0, 0, 0);

 	CvMemStorage* storage = cvCreateMemStorage(0);

 	IplImage* curr = cvQueryFrame(c);
 	CvSize size = cvGetSize(curr);

 	int nx = 640;
 	int ny = 480;
 	float scalex = size.width / nx;
 	float scaley = size.height / ny;

 	CvPoint current, previous;
 	int dx, dy;

 	while (true) {

 		// Query and resize image from camera
 		curr = cvQueryFrame(c);
 		curr_gray_b = cvCreateImage(size, IPL_DEPTH_8U, 1);
 		cvCvtColor(curr, curr_gray_b, CV_RGB2GRAY);
 		curr_gray = cvCreateImage(cvSize(nx, ny), IPL_DEPTH_8U, 1);
 		cvResize(curr_gray_b, curr_gray);

 		if (s  == 0) {

 			// Find the eye
 			seq_l = cvHaarDetectObjects(curr_gray, cascade_l, storage, 1.1, 3, 0, cvSize(x, x));

 			for (i = 0; i < seq_l->total; i++) {
 				rt_l = (CvRect*) cvGetSeqElem(seq_l, 0);
 			}

 			// And crop it to template
 			cropped_l = crop(curr_gray, *rt_l);

 			sizeR = cvSize(
 				std::abs(curr_gray->width  - cropped_l->width)  + 1,
 				std::abs(curr_gray->height - cropped_l->height) + 1
 				);



 		}

  		/* create new image for template matching computation */
 		res = cvCreateImage(sizeR, IPL_DEPTH_32F, 1);
 		cvMatchTemplate(curr_gray, cropped_l, res, CV_TM_SQDIFF);
 		cvMinMaxLoc(res, &minval, &maxval, &minloc, &maxloc, 0);


 		CvRect rect = cvRect(minloc.x, minloc.y, cropped_l->width, cropped_l->height);

 		cropped_l = crop(curr_gray, rect);

 		sizeR = cvSize(
 			std::abs(curr_gray->width  - cropped_l->width)  + 1,
 			std::abs(curr_gray->height - cropped_l->height) + 1
 			);

		/* draw area */
		minloc.x *= scalex;
		minloc.y *= scaley;

 		cvRectangle(curr_gray_b,
 			cvPoint(minloc.x, minloc.y),
 			cvPoint(minloc.x + cropped_l->width*scalex, minloc.y + cropped_l->height*scaley),
 			CV_RGB(255, 0, 0), 1, 0, 0);  

 		current = cvPoint(minloc.x + cropped_l->width*scalex/2, minloc.y + cropped_l->height*scaley/2);
 		if (s == 0) {
 			previous = current;
 		}

 		cvCircle(curr_gray_b, current, 3, CV_RGB(255, 255, 255));

 		dx = current.x - previous.x;
 		dy = current.y - previous.y;

 		movePointer(-dx*8, dy*8);

 		cvShowImage(WINDOW_NAME, curr_gray_b);
 		cvShowImage(WINDOW_TMP, cropped_l);

 		previous = current;
 		cvWaitKey(1);

 		s++;
 	}

 	cvReleaseImage(&curr_gray);
 	cvReleaseImage(&curr_gray_b);
 	cvReleaseImage(&res);
 	cvReleaseImage(&cropped_l);

 	cvReleaseCapture(&c);
 	cvDestroyWindow("Find eye");
 }

