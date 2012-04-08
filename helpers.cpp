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
