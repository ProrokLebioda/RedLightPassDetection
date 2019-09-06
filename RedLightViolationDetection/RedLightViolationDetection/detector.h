#pragma once

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
class Blob;
using namespace cv;
using namespace dnn;
using namespace std;

//function responsible for taking position of pass line
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

class Detector
{
private:
	string classesLine;
	Net net;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;
	string outputFile;
	string kWinName;
	Rect2d myROI;
	Mat croppedFrame;

	float confThreshold = 0.5; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;  // Width of network's input image
	int inpHeight = 416; // Height of network's input image
	vector<string> classes;
public:
	Detector(string classesFile);

	int detectorProgram(CommandLineParser parser);
	string loadClasses(string classesFile);
	bool openVideoOrCam(CommandLineParser parser);
	void loadNetwork();
	void selectUserROI(bool &once);
	void detectionLoop();
	// Remove the bounding boxes with low confidence using non-maxima suppression
	void postprocess(Mat& frame, const vector<Mat>& out);

	// Draw the predicted bounding box
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int i);

	// Get the names of the output layers
	vector<String> getOutputsNames();
};