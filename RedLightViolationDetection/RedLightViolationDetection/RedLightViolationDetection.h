#pragma once

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;


// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// globals
int inX = 0, inY = 0;//starting position for pass line
int outX = 0, outY = 0;//ending position for pass line
bool endLine = false;
void selectUserROI(Rect2d &myROI, Mat frame, Mat& croppedFrame, bool &once, string kWinName);
void mainLoop(Net net, VideoCapture cap, VideoWriter video, Mat frame, Mat croppedFrame, Mat blob, string outputFile, string kWinName);

//function responsible for taking position of pass line
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int i);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;