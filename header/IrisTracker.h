#ifndef __IRIS_TRACKER_IRISTRACKER_H__
#define __IRIS_TRACKER_IRISTRACKER_H__

#include <opencv2/core/core.hpp>

namespace IrisTracker {

	struct TrackerParams
	{
		int Radius_Min;
		int Radius_Max;
		double scaling;
		double lowThres; 
		double highThres;
		bool Conduct_HaarSurrounded;
	};

	const cv::Point2f UNKNOWN_POSITION = cv::Point2f(-1, -1);

	struct findIrisCircle_out {
		cv::Mat_<uchar> mHaarIris;
		cv::Point2f pIris;
		int r_iris;
		
		findIrisCircle_out() : pIris(UNKNOWN_POSITION), r_iris(-1) {}
	};

	bool findIrisCircle(
		const TrackerParams& params,
		const cv::Mat& m,

		findIrisCircle_out& out
		);

} //namespace pupiltracker

#endif//__IRIS_TRACKER_IRISTRACKER_H__
