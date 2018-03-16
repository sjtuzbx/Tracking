#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

const string FILE_NAME = "0001.jpg";

int main(int argc, char* argv[]) {

	Mat img = imread(FILE_NAME, IMREAD_COLOR);
	imshow("Ping Pong", img);
	waitKey(0);
	system("pause");
	return 0;
}