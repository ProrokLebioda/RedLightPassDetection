#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/tracking/tracker.hpp"


class Blob;
using namespace cv;
using namespace dnn;
using namespace std;

//function responsible for taking position of pass line
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

inline double Det(double a, double b, double c, double d)
{
	return a * d - b * c;
}

class MyDetector
{
private:
	string classesLine;
	Net net;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;
	Mat sceneMask;
	string outputFile;
	string kWinName;
	Rect2d carDetectionROI;
	Rect2d trafficLightROI;
	Mat croppedFrame;
	Mat trafficLightFrame;
	vector<Mat> outs;
	Ptr<MultiTracker> multiTracker;
	list<Ptr<Tracker>> singleTrackers;
	list<vector<Point2f>> listOfVectorsOfPointsForTrackers;

	int detectedPasses;

	int iLowH;//Assumed low Hue for red
	int iHighH;//Assumed high Hue for red

	int iLowS;
	int iHighS;

	int iLowV;
	int iHighV;

	Mat frameThresholded;

	bool isRedLight;
	float confThreshold = 0.5; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;  // Width of network's input image
	int inpHeight = 416; // Height of network's input image
	vector<string> classes;
	vector<string> trackerTypes = { "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };
	vector<Rect> trackingBoxes;

	Ptr<Tracker> createTrackerByName(string trackerType);

	bool isIntersecting(Point2f o1, Point2f p1, Point2f o2, Point2f p2);
	list<bool> toSkip;
public:
	MyDetector(string classesFile);

	int detectorProgram(CommandLineParser parser);//class execution
	string loadClasses(string classesFile);
	bool openVideoOrCam(CommandLineParser parser);
	void loadNetwork();
	void selectUserROI(bool &once);
	void detectionLoop();
	// Remove the bounding boxes with low confidence using non-maxima suppression
	void postprocess(Mat& frame, const vector<Mat>& out);
	void setRedLightValues();
	bool detectRedLight();
	// Draw the predicted bounding box
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int i);

	void updateTrackedObjects(Mat &frameCopy );

	// Get the names of the output layers
	vector<String> getOutputsNames();
};