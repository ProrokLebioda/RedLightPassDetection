#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Blob;

class Detector
{
private:
	cv::VideoCapture capVideo;
	struct ScalarColors
	{
		const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
		const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
		const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
		const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
		const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
	}
	SC;

	void MatchCurrentFrameBlobsToExistingBlobs(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs);
	void AddBlobToExistingBlobs(Blob & currentFrameBlob, std::vector<Blob>& existingBlobs, int & intIndex);
	void AddNewBlob(Blob & currentFrameBlob, std::vector<Blob>& existingBlobs);
	double DistanceBetweenPoints(cv::Point point1, cv::Point point2);
	void DrawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
	void DrawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
	void DrawBlobInfoOnImage(std::vector<Blob>& blobs, cv::Mat & imgFrame2Copy);
	bool CheckIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
	void DrawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);

public:
	int Detect(cv::String filename);
	
};