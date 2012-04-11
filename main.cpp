#include "helpers.h"

#define WINDOW_NAME "eye pointer"
#define WINDOW_RIGHT_EYE "Right eye"
#define FACE_SEARCH_WIDTH 80
#define FACE_SEARCH_HEIGHT 60
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 800

int kw, kh;
std::string winname;
int screen_w, screen_h;
cv::Mat bg(screen_w, screen_w, CV_8U);
int crx, cry;
int centerx, centery;


void fun(int rx, int ry, int lx, int ly) {


	crx = (rx - centerx) * kw + screen_w / 2;
	crx = (ry - centery) * kh + screen_h / 2;

	std::cout << "x = " << crx << "\n";
	std::cout << "y = " << cry << "\n";

	cv::circle(bg, cv::Point(crx, cry), 10, CV_RGB(255, 255, 255), -1, 8, 0);
	//cv::imshow(winname, bg);
}

int main() {
	cv::VideoCapture camera(0);
	assert( camera.isOpened() );

	cv::CascadeClassifier faceCascade;
	assert( faceCascade.load("haarcascades/haarcascade_frontalface_alt.xml") );

	cv::CascadeClassifier eyeCascade;
	assert( eyeCascade.load("haarcascades/haarcascade_mcs_eyepair_big.xml") );

	// char key;
	cv::Mat frame;
	// cv::namedWindow( WINDOW_NAME, 1 );
	int threshold = 20;
	// cv::createTrackbar("threshold", WINDOW_NAME, &threshold, 255);
	float faceModX, faceModY;
	// int rx, ry, lx, ly;
	// int counter = 0;
	// int s = 4;
	Pair pupilsRT;
	Pair pupilsRB;
	Pair pupilsLB;
	Pair pupilsLT;
	// int rxs[s], rys[s], lxs[s], lys[s];
	// int frames = 0;

	// zeros(rxs, s);
	// zeros(rys, s);
	// zeros(lxs, s);
	// zeros(lys, s);
	
	camera >> frame;
	faceModX = frame.cols / FACE_SEARCH_WIDTH;
	faceModY = frame.rows / FACE_SEARCH_HEIGHT;

	PupilsFinder finder;
	finder.setFaceCascade(faceCascade);
	finder.setEyeCascade(eyeCascade);
	finder.setResize(FACE_SEARCH_WIDTH, FACE_SEARCH_HEIGHT);
	finder.setFaceMods(faceModX, faceModY);
	finder.setThreshold(threshold);

	std::cout << "Look at RIGHT TOP corner" << std::endl;
	std::cin.get();
	pupilsRT = finder.loop(camera, WINDOW_NAME, 40, 4, true);

	std::cout << "Look at RIGHT BOTTOM corner" << std::endl;
	std::cin.get();
	pupilsRB = finder.loop(camera, WINDOW_NAME, 40, 4, true);

	std::cout << "Look at LEFT BOTTOM corner" << std::endl;
	std::cin.get();
	pupilsLB = finder.loop(camera, WINDOW_NAME, 40, 4, true);

	std::cout << "Look at LEFT TOP corner" << std::endl;
	std::cin.get();
	pupilsLT = finder.loop(camera, WINDOW_NAME, 40, 4, true);

	std::cout << "RT \"right\" eye: " << pupilsRT.first.x << " " << pupilsRT.first.y << std::endl;
	std::cout << "RT \"left\" eye: " << pupilsRT.second.x << " " << pupilsRT.second.y << std::endl << std::endl;

	std::cout << "RB \"right\" eye: " << pupilsRB.first.x << " " << pupilsRB.first.y << std::endl;
	std::cout << "RB \"left\" eye: " << pupilsRB.second.x << " " << pupilsRB.second.y << std::endl << std::endl;

	std::cout << "LB \"right\" eye: " << pupilsLB.first.x << " " << pupilsLB.first.y << std::endl;
	std::cout << "LB \"left\" eye: " << pupilsLB.second.x << " " << pupilsLB.second.y << std::endl << std::endl;

	std::cout << "LT \"right\" eye: " << pupilsLT.first.x << " " << pupilsLT.first.y << std::endl;
	std::cout << "LT \"left\" eye: " << pupilsLT.second.x << " " << pupilsLT.second.y << std::endl << std::endl;


	float kwr = (SCREEN_WIDTH / (abs(pupilsRT.first.x - pupilsLT.first.x)) + SCREEN_WIDTH / (abs(pupilsRB.first.x - pupilsLB.first.x))) / 2;
	float khr = (SCREEN_HEIGHT / (abs(pupilsRT.first.y - pupilsRB.first.y)) + SCREEN_HEIGHT / (abs(pupilsLT.first.y - pupilsLB.first.y))) / 2;

	float kwl = (SCREEN_WIDTH / (abs(pupilsRT.second.x - pupilsLT.second.x)) + SCREEN_WIDTH / (abs(pupilsRB.second.x - pupilsLB.second.x))) / 2;
	float khl = (SCREEN_HEIGHT / (abs(pupilsRT.second.y - pupilsRB.second.y)) + SCREEN_HEIGHT / (abs(pupilsLT.second.y - pupilsLB.second.y))) / 2;
	int centerx = (pupilsRT.first.x + pupilsLT.first.x)/2;
	int centery = (pupilsRT.first.y + pupilsRT.first.y)/2;

	std::cout << "centerx for \"right\" eye: " << centerx << std::endl;
	std::cout << "centery for \"right\" eye: " << centery << std::endl;	

	std::cout << "Kw for \"right\" eye: " << kwr << std::endl;
	std::cout << "Kh for \"right\" eye: " << khr << std::endl;

	std::cout << "Kw for \"left\" eye: " << kwl << std::endl;
	std::cout << "Kh for \"left\" eye: " << khl << std::endl;

	kw = (kwr + kwl) / 2;
	kh = (khr + khl)  / 2;
	screen_w = 800;
	screen_h = 600;

	winname = "Fun";
	cv::namedWindow(winname, 1);

	std::cin.get();

	finder.loopf(camera, WINDOW_NAME, &fun, -1, 4, true);

	// do {
	// 	camera >> frame;

	// 	finder.setThreshold(threshold);	
	// 	pupils = finder.find(frame, true);

	// 	cv::circle(frame, pupils.first, 3, CV_RGB(184, 46, 0));
	// 	cv::circle(frame, pupils.second, 3, CV_RGB(184, 46, 0));

	// 	if (counter == s) counter = 0;

	// 	rxs[counter] = pupils.first.x;
	// 	rys[counter] = pupils.first.y;
	// 	lxs[counter] = pupils.second.x;
	// 	lys[counter] = pupils.second.y;

	// 	if (!(counter % s)) {
	// 		rx = mean(rxs, s);
	// 		ry = mean(rys, s);
	// 		lx = mean(lxs, s);
	// 		ly = mean(lys, s);

	// 		zeros(rxs, s);
	// 		zeros(rys, s);
	// 		zeros(lxs, s);
	// 		zeros(lys, s);
	// 	}

	// 	counter++;
	// 	cv::imshow( WINDOW_NAME, frame);

	// 	key = cv::waitKey(1);
	// 	if (key == ' ') {
	// 		while (true) {
	// 			key = cv::waitKey();
	// 			if (key == ' ') break;
	// 		}
	// 	}

	// 	frames++;
	// } while (key != 27);



	return 0;
}
