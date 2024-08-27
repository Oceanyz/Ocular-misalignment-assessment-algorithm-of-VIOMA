#include "stdafx.h"
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int mostFrequent(int arr[], int n);

struct subtract_out 
{
	vector<vector<float>> copy_A;
	int deleted_row;
};

void subtract(vector<vector<float>> A, int colum1, int colum2, subtract_out& s_out);

vector<vector<float>> delete_row_interation(vector<vector<float>> A, int colum1, int colum2);

cv::Point2f deviation(vector<vector<float>> A, int colum1, int colum2);

float strabismus_degree(float Dev_pixel, float DEp);