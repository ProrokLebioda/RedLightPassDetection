
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

struct hasCrossed{
bool operator()(Vehicle* vehicle) {
	bool state = vehicle->getCrossedState();
	return state; }
};
auto removeVehicles = [&](Vehicle* vehicle) -> bool
{
	bool state = vehicle->getOutOfBounds();
	if (state)
		vehicle->getVehicleTracker().release();
	cout << "Remove Vehicle. Out of Bounds state: " << (bool)state<< endl;
	return state;
};

auto removeWatchdog = [&](Vehicle* vehicle) -> bool
{
	int timer = vehicle->getTimer();
	cout << "I'm sorry,Dave. I have to remove you: " << (int)timer << endl;
	return timer < 0;
};

void MyDetector::drawTrackedObjects(Mat& frameCopy)
{
	int i = 0;

	for (Vehicle* vehicle : vehicles)
	{
		if (vehicle->getOutOfBounds())
			break;
		vehicle->resetTimer();
		Ptr<Tracker> tracker = vehicle->getVehicleTracker();
		Rect2d rect /*= vehicle->getVehicleRect()*/;
		tracker->update(frameCopy, rect);
		vehicle->setVehicleRect(rect);
		int centerX = rect.x + rect.width / 2;
		int bottomCenterY = rect.y + rect.height;
		Point2f bottomCenterOfObjectToTrack(centerX, bottomCenterY);//it's really a center of bottom line
		if (/*((carDetectionROI.x + carDetectionROI.width) > centerOfObjectToTrack.x) && (carDetectionROI.x < centerOfObjectToTrack.x)&&*/carDetectionROI.contains(bottomCenterOfObjectToTrack))
		{
			rectangle(mainFrame, rect, cv::Scalar(0, 0, 255), 2, 1);
			circle(mainFrame, bottomCenterOfObjectToTrack, 5.0, cv::Scalar(0, 255, 0), 2, 1);
			string label = format("No: %d", i);
			cv::putText(mainFrame, label, Point(bottomCenterOfObjectToTrack.x, bottomCenterOfObjectToTrack.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			rectangle(frameCopy, rect, cv::Scalar(0, 0, 0), -2, 1);//paint so that detection will ignore

			if (!vehicle->getVectorOfPointsForTracker().empty())
			{
				vehicle->addPoint(bottomCenterOfObjectToTrack);
				vector<Point2f> points = vehicle->getVectorOfPointsForTracker();

				if (!points.empty())
				{
					Scalar color(0, 0, 255);

					for (int j = 0; j < points.size() - 1; j++)
					{
						cv::line(mainFrame, points[j], points[j + 1], color);
						// add to detected
						Point2f lineO(inX, inY);
						Point2f lineP(outX, outY);
						if (!vehicle->getCrossedState())
						{
							Point2f movementO(points[j].x, points[j].y);
							Point2f movementP(points[j + 1].x, points[j + 1].y);

							if (vehicle->isIntersecting(lineO, lineP, movementO, movementP)&&isRedLight)
							{
								detectedPasses++;
								vehicle->setCrossedState(true);

								cv::Mat imageToSave = mainFrame(vehicle->getVehicleRect());
									
								string filename = "savedViolations/criminal" + to_string(detectedPasses) + ".png";
								imageToSave.convertTo(imageToSave, IMWRITE_PNG_COMPRESSION);

								imwrite(filename, imageToSave);
							}
						}
					}
				}
			}
		}
		else
		{
			if(vehicle->getVehicleRect().y < carDetectionROI.y)
				vehicle->setOutOfBounds(true);
		}
		
		i++;
	}
	list<Vehicle*> tempVehicles;
	for (Vehicle* vehicle : vehicles)
	{
		if (!vehicle->getOutOfBounds())
			tempVehicles.push_back(vehicle);
	}

	if (!tempVehicles.empty())
	{
		vehicles.clear();
		vehicles = tempVehicles;
	}
	
	remove_if(vehicles.begin(), vehicles.end(), removeVehicles);
}

void MyDetector::putEfficiencyInformation()
{
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(mainFrame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	label = format("Detected passes: %d", detectedPasses);
	putText(mainFrame, label, Point(mainFrame.cols - 200, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

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
	detectedPasses(0),
	isRedLight(false),
	lightTimer(0.0)
{
	sceneMask = imread("data/image/00001mask.jpg");
	//multiTracker = cv::MultiTracker::create();
	classesLine = loadClasses(classesFile);
	//iLowH = 165;//Assumed low Hue for red
	//iHighH = 179;//Assumed high Hue for red

	//iLowS = 0;
	//iHighS = 255;

	//iLowV = 0;
	//iHighV = 255;

	iLowH = 0;//Assumed low Hue for red
	iHighH = 27;//Assumed high Hue for red

	iLowS = 12;
	iHighS = 255;

	iLowV = 0;
	iHighV = 255;

	toSkip.clear();

	vehicles.clear();
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
	cap >> mainFrame;
	//resize(mainFrame, mainFrame, Size(mainFrame.cols * 0.50, mainFrame.rows * 0.50), 0, 0, INTER_CUBIC);
	while (cv::waitKey(1) && !endLine)
	{
		putText(mainFrame, "Paint line", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 0, 255), 10);
		cv::circle(mainFrame, cv::Point(inX, inY), 1, Scalar(255, 0, 0));
		cv::circle(mainFrame, cv::Point(outX, outY), 1, Scalar(255, 0, 0));

		imshow(kWinName, mainFrame);

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
		mainFrame.copyTo(frameTemp);

		putText(frameTemp, "Select car detection area", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 0, 255), 10);
		carDetectionROI = selectROI(kWinName, frameTemp);

		mainFrame.copyTo(frameTemp);
		putText(frameTemp, "Select traffic light area", Point(100, 150), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 0, 255), 10);
		trafficLightROI = selectROI(kWinName, frameTemp);
		croppedFrame = mainFrame(carDetectionROI);
		trafficLightFrame = mainFrame(trafficLightROI);

		once = true;

		setRedLightValues();
	}
}

void MyDetector::detectionLoop()
{
	bool once = false;
	Mat mainFrameCopy;
	Mat workFrame;
	
	while (cv::waitKey(30)!=(int)('q'))
	{
		// get frame from the video
		cap >> mainFrame;
		lightTimer++;
		//resize(mainFrame, mainFrame,Size(mainFrame.cols*0.50, mainFrame.rows*0.50), 0, 0, INTER_CUBIC);
		
		
		//uncomment to get  Fake light
		/*if (!once)
			paintFakeStreetLightForCalibration(mainFrame);
		else
			paintFakeStreetLight(mainFrame);*/

		mainFrame.copyTo(mainFrameCopy);
		mainFrame.copyTo(workFrame);
		cv::line(mainFrame, Point(inX, inY), Point(outX, outY), (0, 0, 255),5);
		
		selectUserROI(once);
		if (cv::waitKey(30) == (int)('l'))
			setRedLightValues();

		/*if (waitKey(30) == (int)('t'))
		{*/
			 // adds detected vehicles
		//}

		if (isRedLight=detectRedLight())
			putText(mainFrame, "Red light", Point(trafficLightROI.x + trafficLightFrame.cols + 10, trafficLightROI.y + trafficLightFrame.rows / 2), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 0, 255), 5);
		else
			putText(mainFrame, "Not red light", Point(trafficLightROI.x+trafficLightFrame.cols+10, trafficLightROI.y + trafficLightFrame.rows/2), HersheyFonts::FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 255, 0), 5);
		// Stop the program if we reached end of video
		if (mainFrame.empty())
		{
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			cv::waitKey(3000);
			break;
		}

		
	
		cv::rectangle(mainFrame, carDetectionROI, cv::Scalar(255, 0, 0));
		cv::rectangle(mainFrame, trafficLightROI, cv::Scalar(0, 255, 0));
		//cv::rectangle(frame, carBox, cv::Scalar(0, 0, 255));
		//frameCopy = frameCopy/* & sceneMask*/;
		blobFromImage(mainFrameCopy, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		//vector<Mat> outs;
		net.forward(outs, getOutputsNames());
		if (!vehicles.empty())
		{
			drawTrackedObjects(mainFrameCopy);
		}
		// Remove the bounding boxes with low confidence and paints the prediction boxes
		postprocess(mainFrameCopy, outs);
		updateTrackedObjects(mainFrameCopy);
		for (Vehicle* vehicle : vehicles)
		{
			vehicle->decreaseTimer();
		
		}
		remove_if(vehicles.begin(), vehicles.end(), removeWatchdog);
		/*if (waitKey(30) == (int)('t'))
		{*/
		
		//}
		
		// Draw tracked objects
		
		putEfficiencyInformation();
		// Write the frame with the detection boxes
		mainFrame.convertTo(mainFrame, CV_8U);
		video.write(mainFrame);
		//cv::line(frame, Point(inX, inY), Point(outX, outY), (0, 0, 255), 10);
		imshow(kWinName, mainFrame);
	}
}

int compareBoxes(Rect rectOne, Rect rectTwo)
{
	//top left
	int distancex = (rectTwo.x - rectOne.x) * (rectTwo.x - rectOne.x);
	int distancey = (rectTwo.y - rectOne.y) * (rectTwo.y - rectOne.y);

	double topLeftCalcdistance = sqrt(distancex + distancey);

	//top Right
	distancex = (rectTwo.x+rectTwo.width - rectOne.x + rectOne.width) * (rectTwo.x + rectTwo.width - rectOne.x + rectOne.width);
	distancey = (rectTwo.y - rectOne.y) * (rectTwo.y - rectOne.y);

	double topRightCalcdistance = sqrt(distancex + distancey);

	////bottom Left
	//distancex = (rectTwo.x - rectOne.x) * (rectTwo.x - rectOne.x);
	//distancey = (rectTwo.y+ rectTwo.height - rectOne.y + rectOne.height) * (rectTwo.y + rectTwo.height - rectOne.y+rectOne.height);
	//
	//double bottomLeftCalcdistance = sqrt(distancex + distancey);

	////bottom Right
	//distancex = (rectTwo.x + rectTwo.width - rectOne.x + rectOne.width) * (rectTwo.x + rectTwo.width - rectOne.x + rectOne.width);
	//distancey = (rectTwo.y + rectTwo.height - rectOne.y + rectOne.height) * (rectTwo.y + rectTwo.height - rectOne.y + rectOne.height);

	//double bottomRightCalcdistance = sqrt(distancex + distancey);
	cout << "Top Left distance: " << topLeftCalcdistance << " Top Right distance: " << topRightCalcdistance << endl;
	if (topLeftCalcdistance <= 100.0 || topRightCalcdistance <=100.0 /*||bottomLeftCalcdistance<20.0 || bottomRightCalcdistance<20.0 */)
		return 0;

	return -1;
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

				
				Point2f centerOfObjectToTrack(centerX,centerY);
				if (carDetectionROI.contains(centerOfObjectToTrack) && ((left >= carDetectionROI.x)))
				{
					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}

		
		//Check if not overlapping too much
		for (Rect newBox : boxes)
		{
			bool isDuplicated = false;
			for (Vehicle* vehicle : vehicles)
			{
				if (compareBoxes(newBox, vehicle->getVehicleRect()) == 0)// are the same
				{
					isDuplicated = true;
					cout << "Duplicated Vehicle box\n";
					break;
				}
			}
			for (Rect trBox : trackingBoxes)
			{
				if (compareBoxes(newBox, trBox) == 0)// are the same
				{
					isDuplicated = true;
					cout << "Duplicated trBox\n";
					break;
				}
			}
			if (!isDuplicated)
				trackingBoxes.push_back(newBox);
		}
		
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	if (!boxes.empty())
	{
		NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			Rect box = boxes[idx];
			drawPred(classIds[idx], confidences[idx], box.x, box.y,
				box.x + box.width, box.y + box.height, frame, i);
		}
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
		if (cv::waitKey(30) == 27)
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

	if (mainFrame.empty())
		return false;
	trafficLightFrame = mainFrame(trafficLightROI);
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
	// Specify the tracker type
	string trackerType = "KCF";

	if (!trackingBoxes.empty())
	{
		cout << "Tracking boxes size: " << trackingBoxes.size() << endl;
		for (Rect2d rect : trackingBoxes)
		{
			
			int centerX = rect.x + rect.width / 2;
			int centerY = rect.y + rect.height;
			Point2f centerOfObjectToTrack(centerX, centerY);

			Ptr<TrackerKCF> trackerKCF = TrackerKCF::create();
			trackerKCF->init(frameCopy, rect);

			Vehicle* vehicle = new Vehicle(rect,centerOfObjectToTrack,trackerKCF);

			vehicles.push_back(vehicle);
		}
	}
	trackingBoxes.clear();
}

void MyDetector::paintFakeStreetLightForCalibration(Mat& mF)
{
	Rect2d rect(Point2d(10, 100), Size2d(80, 220));
	rectangle(mF, rect, cv::Scalar(0, 0, 0), -1, 1);
	circle(mF, Point2d(50, 140), 30, Scalar(0, 0, 255), -1, 1);
	circle(mF, Point2d(50, 210), 30, Scalar(0, 255, 255), -1, 1);
	circle(mF, Point2d(50, 280), 30, Scalar(0, 255, 0), -1, 1);
}

void MyDetector::paintFakeStreetLight(Mat& mF)
{
	Rect2d rect(Point2d(10, 100), Size2d(80, 220));
	rectangle(mF, rect, cv::Scalar(0, 0, 0), -1, 1);
	if (lightTimer<240)
	{
		//RED
		circle(mF, Point2d(50, 140), 30, Scalar(0, 0, 255), -1, 1);
		circle(mF, Point2d(50, 210), 30, Scalar(255, 255, 255), 2, 1);
		circle(mF, Point2d(50, 280), 30, Scalar(255, 255, 255), 2, 1);
	}
	else if (lightTimer >= 240 && lightTimer < 280)
	{
		//RED and YELLOW
		circle(mF, Point2d(50, 140), 30, Scalar(0, 0, 255), -1, 1);
		circle(mF, Point2d(50, 210), 30, Scalar(0, 255, 255), -1, 1);
		circle(mF, Point2d(50, 280), 30, Scalar(255, 255, 255), 2, 1);
	}
	else if (lightTimer >= 280 && lightTimer < 460)
	{
		//GREEN
		circle(mF, Point2d(50, 140), 30, Scalar(255, 255, 255), 2, 1);
		circle(mF, Point2d(50, 210), 30, Scalar(255, 255, 255), 2, 1);
		circle(mF, Point2d(50, 280), 30, Scalar(0, 255, 0), -1, 1);
	}
	else if (lightTimer >= 460 && lightTimer<500)
	{
		//YELLOW
		circle(mF, Point2d(50, 140), 30, Scalar(255, 255, 255), 2, 1);
		circle(mF, Point2d(50, 210), 30, Scalar(0, 255, 255), -1, 1);
		circle(mF, Point2d(50, 280), 30, Scalar(255, 255, 255), 2, 1);
	}
	if (lightTimer >= 500)
		lightTimer = 0.0;
	
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