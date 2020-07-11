#include "vehicle.h"

Vehicle::Vehicle(Rect2d _vehicleRect,Point2f _centre, Ptr<Tracker> _tracker)
	: vehicleRect(_vehicleRect),
	vehicleTracker(_tracker),
	hasCrossed(false),
	isOutOfBounds(false)
{
	vectorOfPointsForTracker.clear();
	vectorOfPointsForTracker.push_back(_centre);
	timer = WATCHDOG_VALUE;

}

Vehicle::~Vehicle()
{
	//delete vehicleTracker;
	cout << "Destructor. Crossed state: " << (bool)hasCrossed << endl;
	vehicleTracker.release();
	delete vehicleTracker;
}

void Vehicle::setVehicleRect(Rect2d _vehicleRect)
{
	vehicleRect = _vehicleRect;
}

void Vehicle::setVehicleTracker(Ptr<Tracker> _vehicleTracker)
{
	vehicleTracker = _vehicleTracker;
}

void Vehicle::setVectorOfPointsForTracker(vector<Point2f> _vectorOfPointsForTracker)
{
	vectorOfPointsForTracker = _vectorOfPointsForTracker;
}

void Vehicle::setCrossedState(bool _hasCrossed)
{
	hasCrossed = _hasCrossed;
}

void Vehicle::setOutOfBounds(bool _isOutOfBounds)
{
	isOutOfBounds = _isOutOfBounds;
}

void Vehicle::addPoint(Point2f point)
{
	vectorOfPointsForTracker.push_back(point);
}

//when no info about vehicle updates then we want to decrease timer
void Vehicle::decreaseTimer()
{
	timer--;
}

void Vehicle::resetTimer()
{
	timer = WATCHDOG_VALUE;//set it to the base value
}

bool Vehicle::isIntersecting(Point2f o1, Point2f p1, Point2f o2, Point2f p2)
{
	return doIntersect(o1, p1, o2, p2);
}

bool onSegment(Point2f p, Point2f q, Point2f r)
{

	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
		return true;

	return false;
}

int orientation(Point2f p, Point2f q, Point2f r)
{
	// See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
	// for details of below formula. 
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // colinear 

	return (val > 0) ? 1 : 2; // clock or counterclock wise 
}

bool doIntersect(Point2f p1, Point2f q1, Point2f p2, Point2f q2)
{
	// Find the four orientations needed for general and 
	// special cases 
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case 
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases 
	// p1, q1 and p2 are colinear and p2 lies on segment p1q1 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and q2 are colinear and q2 lies on segment p1q1 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are colinear and p1 lies on segment p2q2 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are colinear and q1 lies on segment p2q2 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases 
}
