// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include "RedLightViolationDetection.h"

using namespace cv;
using namespace dnn;
using namespace std;


int main(int argc, char** argv)
{
	
	CommandLineParser parser(argc, argv, keys);
	parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	// Load names of classes
	string classesFile = "data/yolo/coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Give the configuration and weight files for the model
	String modelConfiguration = "data/yolo/yolov3.cfg";
	String modelWeights = "data/yolo/yolov3_final.weights";

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or a camera stream.
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;
	Mat frameCut;

	try {

		outputFile = "yolo_out_cpp.avi";
		if (parser.has("video"))
		{
			// Open the video file
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("error");
			cap.open(str);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
			outputFile = str;
		}
		// Open the webcaom
		else cap.open(parser.get<int>("device"));

	}
	catch (...) {
		cout << "Could not open the input video stream" << endl;
		return 0;
	}

	// Get the video writer initialized to save the output video
	video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	
	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	//set the callback function for any mouse event
	setMouseCallback(kWinName, CallBackFunc, NULL);
	cap >> frame;
	
	while (cv::waitKey(1) && !endLine)
	{
		putText(frame, "Paint line", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0,Scalar(255,0,255),10);
		cv::circle(frame, cv::Point(inX, inY), 1, Scalar(255, 0, 0));
		cv::circle(frame, cv::Point(outX, outY), 1, Scalar(255, 0, 0));
		
		imshow(kWinName, frame);
		
	}
	


	Mat croppedFrame;
	// Process frames.
	
	mainLoop(net,cap,video,frame,croppedFrame,blob,outputFile,kWinName);

	cap.release();
	video.release();

	cv::waitKey(0);
	return 0;
}

void selectUserROI(Rect2d &myROI,Mat frame, Mat& croppedFrame, bool &once,string kWinName)
{

	if (!once)// Select ROI
	{
		myROI = selectROI(kWinName,frame);
		croppedFrame = frame(myROI);
		once = true;
	}
}

void mainLoop(Net net, VideoCapture cap, VideoWriter video, Mat frame, Mat croppedFrame, Mat blob, string outputFile, string kWinName)
{
	Rect2d myROI;
	bool once = false;
	while (cv::waitKey(1) < 0)
	{
		// get frame from the video
		cap >> frame;
		//resize(frame, frame,Size(frame.cols*0.75,frame.rows*0.75), 0, 0, INTER_CUBIC);
		selectUserROI(myROI, frame, croppedFrame, once,kWinName);
		
		// Stop the program if reached end of video
		if (frame.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			cv::waitKey(3000);
			break;
		}

		cv::rectangle(frame, myROI, cv::Scalar(255, 0, 0));
		blobFromImage(croppedFrame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(croppedFrame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(frame, CV_8U);
		video.write(frame);
		cv::line(frame, Point(inX, inY), Point(outX, outY), (0, 0, 255), 10);
		imshow(kWinName, frame);

	}
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			//1 - bicycle, 2 - car,3 - motorbike, 5 - bus, 7 - truck, 9 - traffic_light 
			if (classIdPoint.x != 1 && classIdPoint.x != 2 && classIdPoint.x != 3 && classIdPoint.x != 5 && classIdPoint.x != 7 && classIdPoint.x != 9)
				continue;
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
				box.x + box.width, box.y + box.height, frame,i);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int i)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, to_string(classId), Point(left, top - ((top - bottom) / 2)),FONT_HERSHEY_SIMPLEX,0.75,Scalar(0,0,255),1);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		inX = x;
		inY = y;
	}
	if (event == EVENT_LBUTTONUP)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		outX = x;
		outY = y;
		endLine = true;
	}
	//else if (event == EVENT_MOUSEMOVE)
	//{
	//	cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

	//}
}
