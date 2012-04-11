#include "helpers.h"

void smoothHist(IntVec & hist, int n) {
	if (n <= 0)
		return;

	IntVec smooth;
	smooth.resize(hist.size());
	while (n) {
		for (int i = 1; i < hist.size() - 1; i++) {
			smooth[i] = hist[i-1] / 4 + hist[i+1] / 4 + hist[i] / 2;
		}
		smooth[0] = smooth[1];
		smooth[hist.size() - 1] = smooth[hist.size() - 2];
		hist = smooth;
		n--;
	}
}

void rectSubImg(cv::Mat & src, cv::Mat & dest, cv::Rect & rect) {
	dest = src
		.rowRange(cv::Range(rect.y, rect.y + rect.height))
		.colRange(cv::Range(rect.x, rect.x + rect.width));
}

void imgHist(cv::Mat & img, IntVec & hHist, IntVec & vHist) {
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
	vHist[0] = vHist[1];
	hHist[0] = hHist[1];
	vHist[vHist.size() - 1] = vHist[vHist.size() - 2];
	hHist[hHist.size() - 1] = hHist[hHist.size() - 2];
}

void drawHist(cv::Mat & image, IntVec & hist, cv::Point start, char xDir, char yDir, int scale) {
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

void splitEyeRect(cv::Rect & eyes, cv::Rect & left, cv::Rect & right) {
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

void findLargestBlob(cv::Mat & src, cv::Rect & largest) {

	cvb::CvBlobs blobs;
	cvb::CvLabel largestL;
	cvb::CvBlob *blob;

	IplImage iplimg = src;
	IplImage *labels = cvCreateImage(src.size(), IPL_DEPTH_LABEL, 1);
	unsigned int result = cvb::cvLabel(&iplimg, labels, blobs);

	largestL = cvb::cvGreaterBlob(blobs);
	blob = blobs[largestL];

	largest = cv::Rect(cv::Point(blob->minx, blob->miny),
		cv::Point(blob->maxx, blob->maxy));

	cvReleaseImage(&labels);
}
int mean(int arr[], int size) {

	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += arr[i];
		if (!arr[i])
			size--;
	}

	return sum / size;
}

void zeros(int arr[], int size) {
	for (int i = 0; i < size; i++) {
		arr[i] = 0;
	}
}


// PupilsFinder
PupilsFinder::PupilsFinder() {

}
void PupilsFinder::setFaceCascade(cv::CascadeClassifier & faceCascade) {
	this->faceCascade = faceCascade;
}
void PupilsFinder::setEyeCascade(cv::CascadeClassifier & eyeCascade) {
	this->eyeCascade = eyeCascade;
}
void PupilsFinder::setThreshold(int threshold) {
	this->threshold = threshold;
}
void PupilsFinder::setResize(int resizeW, int resizeH) {
	this->resizeW = resizeW;
	this->resizeH = resizeH;
}
void PupilsFinder::setFaceMods(float faceModX, float faceModY) {
	this->faceModX = faceModX;
	this->faceModY = faceModY;
}
Pair PupilsFinder::find(cv::Mat & frame, bool drawBorder=false) {

	cv::resize(frame, image, cv::Size(resizeW, resizeH));
	faceCascade.detectMultiScale(image, objects);

	if ( objects.size() ) {
		faceRect = objects[0];

		faceRect.x *= faceModX;
		faceRect.y *= faceModY;
		faceRect.height *= faceModY;
		faceRect.width *= faceModX;

		if (drawBorder) cv::rectangle(frame, faceRect, CV_RGB(255, 0, 0));

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

			//cv::rectangle(frame, leftEye, CV_RGB(255, 255, 0));
			//cv::rectangle(frame, rightEye, CV_RGB(255, 255, 0));

			rectSubImg(frame, rightEyeImage, rightEye);
			rectSubImg(frame, leftEyeImage, leftEye);

			cv::cvtColor(rightEyeImage, rightEyeImage, CV_BGR2GRAY);
			cv::cvtColor(leftEyeImage, leftEyeImage, CV_BGR2GRAY);

			cv::medianBlur(rightEyeImage, rightEyeImage, 7);
			cv::medianBlur(leftEyeImage, leftEyeImage, 7);

			cv::equalizeHist(rightEyeImage, rightEyeImage);
			cv::equalizeHist(leftEyeImage, leftEyeImage);

			cv::threshold(rightEyeImage, rightEyeImage, threshold, 255, CV_THRESH_BINARY_INV);	
			cv::threshold(leftEyeImage, leftEyeImage, threshold, 255, CV_THRESH_BINARY_INV);	

			findLargestBlob(rightEyeImage, rightPupilRect);
			findLargestBlob(leftEyeImage, leftPupilRect);


			if (drawBorder) {
				cv::rectangle(frame, 
					cv::Point(rightEye.x + rightPupilRect.x, 
						rightEye.y + rightPupilRect.y),
					cv::Point(rightEye.x + rightPupilRect.x + rightPupilRect.width, 
						rightEye.y + rightPupilRect.y + rightPupilRect.height),
					CV_RGB(0,0,0));

				cv::rectangle(frame, 
					cv::Point(leftEye.x + leftPupilRect.x, 
						leftEye.y + leftPupilRect.y),
					cv::Point(leftEye.x + leftPupilRect.x + leftPupilRect.width, 
						leftEye.y + leftPupilRect.y + leftPupilRect.height),
					CV_RGB(0,0,0));
			}

			r = cv::Point(
				(rightEye.x + rightPupilRect.x + 
					rightEye.x + rightPupilRect.x + rightPupilRect.width) / 2, 
				(rightEye.y + rightPupilRect.y + 
					rightEye.y + rightPupilRect.y + rightPupilRect.height) / 2);

			l = cv::Point(
				(leftEye.x + leftPupilRect.x + 
					leftEye.x + leftPupilRect.x + leftPupilRect.width) / 2,
				(leftEye.y + leftPupilRect.y + 
					leftEye.y + leftPupilRect.y + leftPupilRect.height) / 2);


			// return pair(cv::Point(3,2), cv::Point(3, 2) );	
			return Pair(r, l);
		}
	}
}
Pair PupilsFinder::loop(cv::VideoCapture & camera, std::string wname, int iterations=-1, 
	int aprxIterations=4, bool drawBorder=false) {

	cv::namedWindow( wname, 1 );
	cv::createTrackbar("threshold", wname, &threshold, 255);

	counter = 0;
	frames = 0;

	int rxs[aprxIterations], 
		rys[aprxIterations], 
		lxs[aprxIterations], 
		lys[aprxIterations];

	zeros(rxs, aprxIterations);
	zeros(rys, aprxIterations);
	zeros(lxs, aprxIterations);
	zeros(lys, aprxIterations);

	camera >> frame;

	do {
		camera >> frame;
		pupils = find(frame, drawBorder);

		cv::circle(frame, pupils.first, 3, CV_RGB(184, 46, 0));
		cv::circle(frame, pupils.second, 3, CV_RGB(184, 46, 0));

		if (counter == aprxIterations) counter = 0;

		rxs[counter] = pupils.first.x;
		rys[counter] = pupils.first.y;
		lxs[counter] = pupils.second.x;
		lys[counter] = pupils.second.y;

		if (!(counter % aprxIterations)) {
			rx = mean(rxs, aprxIterations);
			ry = mean(rys, aprxIterations);
			lx = mean(lxs, aprxIterations);
			ly = mean(lys, aprxIterations);

			zeros(rxs, aprxIterations);
			zeros(rys, aprxIterations);
			zeros(lxs, aprxIterations);
			zeros(lys, aprxIterations);
		}

		counter++;
		cv::imshow( wname, frame);

		key = cv::waitKey(1);
		if (key == ' ') {
			while (true) {
				key = cv::waitKey();
				if (key == ' ') break;
			}
		}

		if (iterations != -1)
			frames++;
	} while ((key != 27) && ((frames <= iterations) || (iterations == -1)));

	return pupils;
}

void PupilsFinder::loopf(cv::VideoCapture & camera, std::string wname, void (*f)(int, int, int, int),
	int iterations=-1, int aprxIterations=4,  bool drawBorder=false) {

	cv::namedWindow( wname, 1 );
	cv::createTrackbar("threshold", wname, &threshold, 255);

	counter = 0;
	frames = 0;

	int rxs[aprxIterations], 
		rys[aprxIterations], 
		lxs[aprxIterations], 
		lys[aprxIterations];

	zeros(rxs, aprxIterations);
	zeros(rys, aprxIterations);
	zeros(lxs, aprxIterations);
	zeros(lys, aprxIterations);

	camera >> frame;

	do {
		camera >> frame;
		pupils = find(frame, drawBorder);

		cv::circle(frame, pupils.first, 3, CV_RGB(184, 46, 0));
		cv::circle(frame, pupils.second, 3, CV_RGB(184, 46, 0));

		if (counter == aprxIterations) counter = 0;

		rxs[counter] = pupils.first.x;
		rys[counter] = pupils.first.y;
		lxs[counter] = pupils.second.x;
		lys[counter] = pupils.second.y;

		if (!(counter % aprxIterations)) {
			rx = mean(rxs, aprxIterations);
			ry = mean(rys, aprxIterations);
			lx = mean(lxs, aprxIterations);
			ly = mean(lys, aprxIterations);

			zeros(rxs, aprxIterations);
			zeros(rys, aprxIterations);
			zeros(lxs, aprxIterations);
			zeros(lys, aprxIterations);
		}

		(*f) (rx, ry, lx, ly);

		counter++;
		cv::imshow( wname, frame);

		key = cv::waitKey(1);
		if (key == ' ') {
			while (true) {
				key = cv::waitKey();
				if (key == ' ') break;
			}
		}

		if (iterations != -1)
			frames++;
	} while ((key != 27) && ((frames <= iterations) || (iterations == -1)));
}

Mover::Mover(int kw, int kh) {
	this->kw = kw;
	this->kh = kh;
	this->wname = "Fun";
	this->screen_w = 1280;
	this->screen_h = 800;
	cv::namedWindow(this->wname, 1);
	cv::resize(bg, bg, cv::Size(screen_w, screen_h));
	bg.zeros(screen_w, screen_h, CV_8U);
}

void Mover::fun(int rx, int ry, int lx, int ly) {

	cry = (ry*kh + ly*kh) / 2;
	crx = (rx*kw + ry*kw) / 2;

	cv::circle(bg, cv::Point(crx, cry), 10, CV_RGB(255, 255, 255), -1, 8, 0);
	cv::imshow(wname, bg);
}