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

using namespace std;
using namespace cv;

#define WATCHDOG_VALUE 15

inline double Det(double a, double b, double c, double d)
{
	return (a * d) - (b * c);
}

// Given three colinear points p, q, r, the function checks if 
// point q lies on line segment 'pr' 
bool onSegment(Point2f p, Point2f q, Point2f r);


// To find orientation of ordered triplet (p, q, r). 
// The function returns following values 
// 0 --> p, q and r are colinear 
// 1 --> Clockwise 
// 2 --> Counterclockwise 
int orientation(Point2f p, Point2f q, Point2f r);

// The main function that returns true if line segment 'p1q1' 
// and 'p2q2' intersect. 
bool doIntersect(Point2f p1, Point2f q1, Point2f p2, Point2f q2);

class Vehicle
{
public:
	Vehicle(Rect2d _vehicleRect, Point2f _centre, Ptr<Tracker> _tracker);
	~Vehicle();
	Rect2d getVehicleRect() { return vehicleRect; };
	Ptr<Tracker> getVehicleTracker() { return vehicleTracker; };
	vector<Point2f> getVectorOfPointsForTracker() { return vectorOfPointsForTracker; };
	bool getCrossedState() { return hasCrossed; };
	bool getOutOfBounds() { return isOutOfBounds; };
	int getTimer() { return timer; };

	void setVehicleRect(Rect2d _vehicleRect);
	void setVehicleTracker(Ptr<Tracker> _vehicleTracker);
	void setVectorOfPointsForTracker(vector<Point2f> _vectorOfPointsForTracker); 
	void setCrossedState(bool _hasCrossed);
	void setOutOfBounds(bool _isOutOfBounds);
	void addPoint(Point2f point);
	void decreaseTimer();
	void resetTimer();

	bool isIntersecting(Point2f o1, Point2f p1, Point2f o2, Point2f p2);
private:
	Rect2d vehicleRect;
	vector<Point2f> vectorOfPointsForTracker;
	bool hasCrossed; //set to true after object passed line to skip counting it again
	bool isOutOfBounds;
	Ptr<Tracker> vehicleTracker;
	int timer;
};