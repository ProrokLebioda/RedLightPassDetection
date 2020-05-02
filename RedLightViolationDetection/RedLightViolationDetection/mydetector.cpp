
#pragma once
#include "mydetector.h"
//#include "blob.h"
//#include <conio.h>

// globals for mouse callback
int inX = 0, inY = 0;//starting position for pass line
int outX = 0, outY = 0;//ending position for pass line
bool endLine = false;

Ptr<Tracker> MyDetector::createTrackerByName(string trackerType)
{
	Ptr<Tracker> tracker;
	if (trackerType == trackerTypes[0])
		tracker = TrackerBoosting::create();
	else if (trackerType == trackerTypes[1])
		tracker = TrackerMIL::create();
	else if (trackerType == trackerTypes[2])
		tracker = TrackerKCF::create();
	else if (trackerType == trackerTypes[3])
		tracker = TrackerTLD::create();
	else if (trackerType == trackerTypes[4])
		tracker = TrackerMedianFlow::create();
	else if (trackerType == trackerTypes[5])
		tracker = TrackerGOTURN::create();
	else if (trackerType == trackerTypes[6])
		tracker = TrackerMOSSE::create();
	else if (trackerType == trackerTypes[7])
		tracker = TrackerCSRT::create();
	else {
		cout << "Incorrect tracker name" << endl;
		cout << "Available trackers are: " << endl;
		for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); ++it)
			std::cout << " " << *it << endl;
	}
	return tracker;
}

void MyDetector::drawTrackedObjects(Mat& frameCopy)
{
	list<Ptr<Tracker>>::iterator itTracker;
	itTracker = singleTrackers.begin();

	list<vector<Point2f>>::iterator itListOfVectorsOfPointsForTrackers;
	itListOfVectorsOfPointsForTrackers = listOfVectorsOfPointsForTrackers.begin();

	list<bool>::iterator itToSkip;
	itToSkip = toSkip.begin();
	for (int i = 0; i < singleTrackers.size(); i++)
	{
		itTracker = next(singleTrackers.begin(), i);
		Ptr<Tracker> tracker = *itTracker;
		Rect2d rect;
		tracker->update(frame, rect);

		itListOfVectorsOfPointsForTrackers = next(listOfVectorsOfPointsForTrackers.begin(), i);

		itToSkip = next(toSkip.begin(), i);

		int centerX = rect.x + rect.width / 2;
		int centerY = rect.y + rect.height;
		Point2f centerOfObjectToTrack(centerX, centerY);

		if (((carDetectionROI.x + carDetectionROI.width) > centerOfObjectToTrack.x) && (carDetectionROI.x < centerOfObjectToTrack.x))
		{

			rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, 1);
			string label = format("No: %d", i);
			cv::putText(frame, label, Point(centerOfObjectToTrack.x, centerOfObjectToTrack.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			rectangle(frameCopy, rect, cv::Scalar(0, 0, 255), -2, 1);//paint so that detection will ignore

			if (!itListOfVectorsOfPointsForTrackers->empty())
			{
				itListOfVectorsOfPointsForTrackers->push_back(centerOfObjectToTrack);
				vector<Point2f> points = *itListOfVectorsOfPointsForTrackers;

				if (!points.empty())
				{
					Scalar color(0, 0, 255);
					for (int j = 0; j < points.size() - 1; j++)
					{
						cv::line(frame, points[j], points[j + 1], color);
						// add to detected
						Point2f lineO(inX, inY);
						Point2f lineP(outX, outY);
						if (*itToSkip != true)
						{
							Point2f movementO(points[j].x, points[j].y);
							Point2f movementP(points[j + 1].x, points[j + 1].y);

							if (isIntersecting(lineO, lineP, movementO, movementP))
							{
								detectedPasses++;
								*itToSkip = true;
							}
						}
					}
				}
			}
		}
		else
		{
			singleTrackers.erase(itTracker);
			listOfVectorsOfPointsForTrackers.erase(itListOfVectorsOfPointsForTrackers);
			toSkip.erase(itToSkip);
		}
	}
}

void MyDetector::putEfficiencyInformation()
{
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	label = format("Detected passes: %d", detectedPasses);
	putText(frame, label, Point(frame.cols - 200, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

}

bool MyDetector::isIntersecting(Point2f o1, Point2f p1, Point2f o2, Point2f p2)
{
	//Point2f x = o2 - o1;
	//Point2f d1 = p1 - o1;
	//Point2f d2 = p2 - o2;
	
	//double cross = d1.x * d2.y - d1.y * d2.x;
	//if (std::abs(cross) < /*EPS*/1e-8)
	//	return false;
	float ixOut;
	float iyOut;

	float detL1 = Det(o1.x, o1.y, p1.x, p1.y);
	float detL2 = Det(o2.x, o2.y, p2.x, p2.y);
	
	float x1mx2 = o1.x - p1.x;
	float x3mx4 = o2.x - p2.x;
	float y1my2 = o1.y - p1.y;
	float y3my4 = o2.y - p2.y;

	float xnom = Det(detL1, x1mx2, detL2, x3mx4);
	float ynom = Det(detL1, y1my2, detL2, y3my4);
	float denom = Det(x1mx2, y1my2, x3mx4, y3my4);
	if (denom == 0.0)//Lines don't seem to cross
	{
		ixOut = NAN;
		iyOut = NAN;
		return false;
	}

	ixOut = xnom / denom;
	iyOut = ynom / denom;
	if (!isfinite(ixOut) || !isfinite(iyOut)) //Probably a numerical issue
		return false;

	return true; //All OK
}

MyDetector::MyDetector(string classesFile) :
	confThreshold(0.5), nmsThreshold(0.4),
	inpWidth(416), inpHeight(416),
	detectedPasses(0)
{
	sceneMask = imread("data/image/00001mask.jpg");
	//multiTracker = cv::MultiTracker::create();
	classesLine = loadClasses(classesFile);
	iLowH = 165;//Assumed low Hue for red
	iHighH = 179;//Assumed high Hue for red

	iLowS = 0;
	iHighS = 255;

	iLowV = 0;
	iHighV = 255;
	toSkip.clear();
}

int MyDetector::detectorProgram(CommandLineParser parser)
{
	loadNetwork();

	if (!openVideoOrCam(parser))
		return 0;

	// Get the video writer initialized to save the output video
	video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	
	// Create a window
	kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	//set the callback function for any mouse event
	setMouseCallback(kWinName, CallBackFunc, NULL);
	cap >> frame;

	while (cv::waitKey(1) && !endLine)
	{
		putText(frame, "Paint line", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 0, 255), 10);
		cv::circle(frame, cv::Point(inX, inY), 1, Scalar(255, 0, 0));
		cv::circle(frame, cv::Point(outX, outY), 1, Scalar(255, 0, 0));

		imshow(kWinName, frame);

	}

	detectionLoop();

	cap.release();
	video.release();

	cv::waitKey(0);
	return 0;
}

string MyDetector::loadClasses(string classesFile)
{
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);
	return line;
}

bool MyDetector::openVideoOrCam(CommandLineParser parser)
{
	string str;
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
		// Open the webcam
		else cap.open(parser.get<int>("device"));
		return true;
	}
	catch (...) {
		cout << "Could not open the input video stream" << endl;
		return false;
	}
	return false;
}


void MyDetector::loadNetwork()
{
	// Give the configuration and weight files for the model
	String modelConfiguration = "data/yolo/yolov3.cfg";
	String modelWeights = "data/yolo/yolov3_final.weights";

	// Load the network
	net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL);
}

void MyDetector::selectUserROI(bool &once)
{
	if (!once)// Select ROI
	{
		Mat frameTemp;
		frame.copyTo(frameTemp);

		putText(frameTemp, "Select car detection area", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 0, 255), 10);
		carDetectionROI = selectROI(kWinName, frameTemp);

		frame.copyTo(frameTemp);
		putText(frameTemp, "Select traffic light area", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 0, 255), 10);
		trafficLightROI = selectROI(kWinName, frameTemp);
		croppedFrame = frame(carDetectionROI);
		trafficLightFrame = frame(trafficLightROI);

		once = true;

		setRedLightValues();
	}
}

void MyDetector::detectionLoop()
{
	bool once = false;
	Mat frameCopy;
	
	while (waitKey(30)!=(int)('q'))
	{
		// get frame from the video
		cap >> frame;
		frame.copyTo(frameCopy);
		cv::line(frame, Point(inX, inY), Point(outX, outY), (0, 0, 255), 10);
		//resize(frame, frame,Size(frame.cols*0.75,frame.rows*0.75), 0, 0, INTER_CUBIC);
		selectUserROI(once);
		if (waitKey(30) == (int)('l'))
			setRedLightValues();

		/*if (waitKey(30) == (int)('t'))
		{*/
			updateTrackedObjects(frameCopy);
		//}

		if (detectRedLight())
			putText(frame, "Red light", Point(trafficLightROI.x + trafficLightFrame.cols + 10, trafficLightROI.y + trafficLightFrame.rows / 2), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 0, 255), 5);
		else
			putText(frame, "Not red light", Point(trafficLightROI.x+trafficLightFrame.cols+10, trafficLightROI.y + trafficLightFrame.rows/2), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 255, 0), 5);
		// Stop the program if we reached end of video
		if (frame.empty())
		{
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			cv::waitKey(3000);
			break;
		}

		// Draw tracked objects
		if (!singleTrackers.empty())
		{
			drawTrackedObjects(frameCopy);			
		}
	
		cv::rectangle(frame, carDetectionROI, cv::Scalar(255, 0, 0));
		cv::rectangle(frame, trafficLightROI, cv::Scalar(0, 255, 0));
		//cv::rectangle(frame, carBox, cv::Scalar(0, 0, 255));
		frameCopy = frameCopy/* & sceneMask*/;
		blobFromImage(frameCopy, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		//vector<Mat> outs;
		net.forward(outs, getOutputsNames());

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		putEfficiencyInformation();
		// Write the frame with the detection boxes
		frame.convertTo(frame, CV_8U);
		video.write(frame);
		//cv::line(frame, Point(inX, inY), Point(outX, outY), (0, 0, 255), 10);
		imshow(kWinName, frame);
	}
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void MyDetector::postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	trackingBoxes.clear();
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
			if (classIdPoint.x != 1 && classIdPoint.x != 2 && classIdPoint.x != 3 && classIdPoint.x != 5 && classIdPoint.x != 7 /*&& classIdPoint.x != 9*/)
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
				Point2f centerOfObjectToTrack(centerX,centerY);
				//if (width>200&&height>200)
				if (carDetectionROI.contains(centerOfObjectToTrack) && ((left >= carDetectionROI.x)))
				{
					//Check if not overlapping too much
					for (Rect trBox : trackingBoxes)
					{
						
					}
					trackingBoxes.push_back(Rect(left, top, width, height));
				}
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
			box.x + box.width, box.y + box.height, frame, i);
	}
}

void MyDetector::setRedLightValues()
{
	//trafficLightROI
	namedWindow("Control", WINDOW_NORMAL);

	createTrackbar("LowH", "Control", &iLowH, 179);//Hue 0-179
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255);//Saturation 0-255
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255);//Value 0-255
	createTrackbar("HighV", "Control", &iHighV, 255);

	while (true)
	{
		Mat frameHSV;
		cvtColor(trafficLightFrame, frameHSV, COLOR_BGR2HSV);
		//Mat frameThresholded;
		inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), frameThresholded);

		//morphological opening (remove small objects from the foreground)
		erode(frameThresholded, frameThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(frameThresholded, frameThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		imshow("Thresholded Frame", frameThresholded);
		imshow("Original Frame", trafficLightFrame);
		if (waitKey(30) == 27)
		{
			setWindowProperty("Thresholded Frame", cv::WindowPropertyFlags::WND_PROP_VISIBLE, 0.0);
			setWindowProperty("Original Frame", cv::WindowPropertyFlags::WND_PROP_VISIBLE, 0.0);
			cout << "ESC key is pressed by user" << endl;
			break;
		}
	}
}

bool MyDetector::detectRedLight()
{
	int area=0;
	int detectedHueCount=0;
	double percentage=0.0;

	trafficLightFrame = frame(trafficLightROI);
	Mat frameHSV;
	cvtColor(trafficLightFrame, frameHSV, COLOR_BGR2HSV);
	//Mat frameThresholded;
	inRange(frameHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), frameThresholded);
	//morphological opening (remove small objects from the foreground)
	erode(frameThresholded, frameThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(frameThresholded, frameThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	for (int row = 0; row < frameThresholded.rows; row++)
	{
		for (int col= 0; col < frameThresholded.cols; col++)
		{
			uchar pixelGrayValue = frameThresholded.at<uchar>(row, col);
			area++;
			if (pixelGrayValue > 0)
				detectedHueCount++;
		}
	}
	if (detectedHueCount > 0)
	{
		percentage = ((double)detectedHueCount*100) / area;
		if (percentage > 2.0)
			return true;
		else
			return false;
	}
	else
	{
		return false;
	}
	
}

// Draw the predicted bounding box
void MyDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, int i)
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
	putText(frame, to_string(classId), Point(left, top - ((top - bottom) / 2)), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 1);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

void MyDetector::updateTrackedObjects(Mat &frameCopy)
{
	if (!trackingBoxes.empty())
	{
		// Specify the tracker type
		string trackerType = "KCF";
		for (Rect2d rect : trackingBoxes)
		{

			int centerX = rect.x + rect.width / 2;
			int centerY = rect.y + rect.height;
			Point2f centerOfObjectToTrack(centerX, centerY);

			Ptr<TrackerKCF> trackerKCF = TrackerKCF::create();
			trackerKCF->init(frameCopy, rect);
			singleTrackers.push_back(trackerKCF);
			vector<Point2f> pointsForTracker;
			pointsForTracker.push_back(centerOfObjectToTrack);
			listOfVectorsOfPointsForTrackers.push_back(pointsForTracker);
			toSkip.push_back(false);
		}
	}
}

// Get the names of the output layers
vector<String> MyDetector::getOutputsNames()
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