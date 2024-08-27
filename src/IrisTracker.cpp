#include "stdafx.h"
#include "IrisTracker.h"
#include "PupilTracker.h"
#include "cvx.h"
#include "Masek.h"
#include "ImageUtility.h"
#include <iostream>
#include <boost/foreach.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tbb/tbb.h>
#include "FindPupilCircleNew.h"
#include "ImageUtility.h"
#include "GammaCorrection.h"
using namespace std;
using namespace cv;



class HaarSurroundFeature
{
public:
	HaarSurroundFeature(int r1, int r2) : r_inner(r1), r_outer(r2)
	{
		//  _________________
		// |        -ve      |
		// |     _______     |
		// |    |   +ve |    |
		// |    |   .   |    |
		// |    |_______|    |
		// |         <r1>    |
		// |_________<--r2-->|

		// Number of pixels in each part of the kernel
		int count_inner = r_inner*r_inner;
		int count_outer = r_outer*r_outer - r_inner*r_inner;

		val_inner = 1.0 / (r_inner*r_inner);
		val_outer = -val_inner*count_inner / count_outer;

	}

	double val_inner, val_outer;
	int r_inner, r_outer;
};


bool IrisTracker::findIrisCircle(
	const IrisTracker::TrackerParams& params,
	const cv::Mat& m,
	IrisTracker::findIrisCircle_out& out
	)
{

	cv::Mat_<uchar> srcEye;
	// --------------------
	// Convert to greyscale
	// --------------------
	{
		// Pick one channel if necessary, and crop it to get rid of borders
		if (m.channels() == 1)
		{
			srcEye = m;
		}
		else if (m.channels() == 3)
		{
			cv::cvtColor(m, srcEye, cv::COLOR_BGR2GRAY);
		}
		else if (m.channels() == 4)
		{
			cv::cvtColor(m, srcEye, cv::COLOR_BGRA2GRAY);
		}
		else
		{
			throw std::runtime_error("Unsupported number of channels");
		}
	}

	//原图像gamma变换
	Mat mEye(srcEye.rows, srcEye.cols, srcEye.type());
	GammaCorrection(srcEye, mEye, 0.5);  //srcEye为原始眼睛图象；mEye为gamma变换后眼睛图象

	// -----------------------
	// *****step1: Find best haar response通过Haar-surrounded feature寻找iris的缩小区域   (可选)
	// -----------------------

	//             _____________________
	//            |         Haar kernel |
	//            |                     |
	//  __________|______________       |
	// | Image    |      |       |      |
	// |    ______|______|___.-r-|--2r--|
	// |   |      |      |___|___|      |
	// |   |      |          |   |      |
	// |   |      |          |   |      |
	// |   |      |__________|___|______|
	// |   |    Search       |   |
	// |   |    region       |   |
	// |   |                 |   |
	// |   |_________________|   |
	// |                         |
	// |_________________________|
	//
	IplImage *Haarimg;
	cv::Point2f pHaarIris;
	int haarRadius = 0;
	if(params.Conduct_HaarSurrounded)  //true则执行Haar搜索
	{
		cv::Mat_<int32_t> mEyeIntegral;
		int padding = 2 * params.Radius_Max;
		//SECTION("Integral image", log)
		{
			cv::Mat mEyePad;
			// Need to pad by an additional 1 to get bottom & right edges.
			cv::copyMakeBorder(mEye, mEyePad, padding, padding, padding, padding, cv::BORDER_REPLICATE);
			cv::integral(mEyePad, mEyeIntegral);
		}

		//SECTION("Haar responses", log)
		{
			const int rstep = 2;
			const int ystep = 4;
			const int xstep = 4;

			double minResponse = std::numeric_limits<double>::infinity();

			for (int r = params.Radius_Min; r < params.Radius_Max; r += rstep)
			{
				// Get Haar feature
				int r_inner = r;
				int r_outer = 3 * r;  //2 * r
				HaarSurroundFeature f(r_inner, r_outer);

				// Use TBB for rows
				std::pair<double, cv::Point2f> minRadiusResponse = tbb::parallel_reduce(
					tbb::blocked_range<int>(0, (mEye.rows - r - r - 1) / ystep + 1, ((mEye.rows - r - r - 1) / ystep + 1) / 8),
					std::make_pair(std::numeric_limits<double>::infinity(), UNKNOWN_POSITION),
					[&](tbb::blocked_range<int> range, const std::pair<double, cv::Point2f>& minValIn) -> std::pair<double, cv::Point2f>
				{
					std::pair<double, cv::Point2f> minValOut = minValIn;
					for (int i = range.begin(), y = r + range.begin()*ystep; i < range.end(); i++, y += ystep)
					{
						//            ?        ?                    // row1_outer.|         |  p00._____________________.p01
						//            |         |     |         Haar kernel |
						//            |         |     |                     |
						// row1_inner.|         |     |   p00._______.p01   |
						//            |-padding-|     |      |       |      |
						//            |         |     |      | (x,y) |      |
						// row2_inner.|         |     |      |_______|      |
						//            |         |     |   p10'       'p11   |
						//            |         |     |                     |
						// row2_outer.|         |     |_____________________|
						//            |         |  p10'                     'p11
						//            ?        ?
						int* row1_inner = mEyeIntegral[y + padding - r_inner];
						int* row2_inner = mEyeIntegral[y + padding + r_inner + 1];
						int* row1_outer = mEyeIntegral[y + padding - r_outer];
						int* row2_outer = mEyeIntegral[y + padding + r_outer + 1];

						int* p00_inner = row1_inner + r + padding - r_inner;
						int* p01_inner = row1_inner + r + padding + r_inner + 1;
						int* p10_inner = row2_inner + r + padding - r_inner;
						int* p11_inner = row2_inner + r + padding + r_inner + 1;

						int* p00_outer = row1_outer + r + padding - r_outer;
						int* p01_outer = row1_outer + r + padding + r_outer + 1;
						int* p10_outer = row2_outer + r + padding - r_outer;
						int* p11_outer = row2_outer + r + padding + r_outer + 1;

						for (int x = r; x < mEye.cols - r; x += xstep)
						{
							int sumInner = *p00_inner + *p11_inner - *p01_inner - *p10_inner;
							int sumOuter = *p00_outer + *p11_outer - *p01_outer - *p10_outer - sumInner;

							double response = f.val_inner * sumInner + f.val_outer * sumOuter;

							if (response < minValOut.first)
							{
								minValOut.first = response;
								minValOut.second = cv::Point(x, y);
							}

							p00_inner += xstep;
							p01_inner += xstep;
							p10_inner += xstep;
							p11_inner += xstep;

							p00_outer += xstep;
							p01_outer += xstep;
							p10_outer += xstep;
							p11_outer += xstep;
						}
					}
					return minValOut;
				},
					[](const std::pair<double, cv::Point2f>& x, const std::pair<double, cv::Point2f>& y) -> std::pair<double, cv::Point2f>
				{
					if (x.first < y.first)
						return x;
					else
						return y;
				}
				);

				if (minRadiusResponse.first < minResponse)
				{
					minResponse = minRadiusResponse.first;
					// Set return values
					pHaarIris = minRadiusResponse.second;
					haarRadius = r;
				}
			}
		}//section over
		// Paradoxically, a good Haar fit won't catch the entire pupil, so expand it a bit
		haarRadius = (int)(haarRadius * sqrt(2.0));

		// ---------------------------
		// iris ROI around Haar point
		// ---------------------------
		cv::Rect roiHaarIris = pupiltracker::cvx::roiAround(cv::Point(pHaarIris.x, pHaarIris.y), haarRadius);
		cv::Mat_<uchar> mHaarIris;

		pupiltracker::cvx::getROI(mEye, mHaarIris, roiHaarIris);
		out.mHaarIris = mHaarIris;

		//cv::imshow("irisregion",mHaarIris);
		//cv::waitKey(0);
		Haarimg = (IplImage *)&IplImage(mHaarIris);
	}
	else
	{
		Haarimg = (IplImage *)&IplImage(mEye);    //如果不执行haar，则瞳孔区域搜索仍在mEye上进行
	}

	//*****step2: 由于瞳孔区域比较显著，所以先把瞳孔找出来，进一步提取iris roi，缩小搜索区域
	//PUPIL INPUTS	
	float nScale = 1.0;
	const int speed_m = 1;// Default 1
	int alpha = 20; // Alpha value for contrast threshold
	// Setup the parameters to avoid that noise caused by reflections and 
	// eyelashes covers the pupil
	double ratio4Circle = 0.9;
	// Initialize for Closing and Opening process
	int closeItr = 2;//dilate->erode
	int openItr = 3;//erode->dilate
	double norm = 256.0;//
	const int rPupilMax = (int)(45 * nScale);// Maximum radius of pupil's circle 
	int pupilCircle[6] = { 0 };
	FindPupilCircleNew::doDetect(Haarimg, rPupilMax, ratio4Circle, closeItr, openItr, speed_m, alpha, norm, nScale, pupilCircle);

	CvPoint xyPupil;
	xyPupil.x = pupilCircle[0];
	xyPupil.y = pupilCircle[1];
	int rPupil = pupilCircle[2];
	// Draw the pupil circle on  Haarimg *****************************************************************************************************************
	//cvCircle(Haarimg, cvPoint(xyPupil.x, xyPupil.y), rPupil, CV_RGB(255, 255, 255), 1, 8);
	//ImageUtility::showImage("Pupil Circle", Haarimg);
	
	//由于瞳孔区域一般比较完整，但是Haar操作有可能iris区域不完整，所以将mHaarImage上的瞳孔检测结果映射回原图，在原图上提取虹膜ROI
	//calculate the origin pupil center coordinates on the eye image 
	if (params.Conduct_HaarSurrounded)
	{
		xyPupil.x = pHaarIris.x - haarRadius + xyPupil.x;
		xyPupil.y = pHaarIris.y - haarRadius + xyPupil.y;
	}
	else
	{
	}
	// Draw the pupil circle on mEye *****************************************************************************************************************
	//cv::circle(mEye, xyPupil, rPupil, Scalar(0, 255, 0), 3, 8, 0);// circle outline
	//cv::imshow("pupil circle", mEye);

	//ROI for detecting the iris circle using pupil information on the mEye
	IplImage *Eyeimg = (IplImage *)&IplImage(mEye);
	ImageUtility *imgUtil = NULL;
	ImageUtility::SETVALUE setVal = imgUtil->setImage(Eyeimg, xyPupil, rPupil, params.Radius_Max, params.Radius_Max);	//82 is the best for video images, previous 80
	IplImage* setImg = NULL;
	setImg = imgUtil->getROIImage(Eyeimg, setVal.rect.x, setVal.rect.width, setVal.rect.y, setVal.rect.height);
	if (setImg == NULL)
	{
		cout << "Failed to load the file" << endl;
		return -1;
	}
	//ImageUtility::showImage("Iris ROI", setImg);  //画出虹膜ROI区域*********************************************************************************

	//*****step3:在iris ROI上通过Hough Transform方法检测iris
	// Values returned by the Hough Transform
	int rowIris, colIris, rIris;
	cv::Point2f pIris;
	int r_Iris;
	//Initialize
	pIris.x = 1; //x
	pIris.y = 1; //y
	r_Iris = 1; //radius

	// Convert the IplImage to IMAGE
	Masek::IMAGE * irisImg = ImageUtility::convertIplToImage(setImg);

	// Find the iris boundaries for both video and still images
	Masek *masek = NULL;
	masek->findcircle(irisImg, params.Radius_Min, params.Radius_Max, params.scaling, 2, params.highThres, params.lowThres, 1.00, 0.00, &rowIris, &colIris, &rIris);
	delete masek;
	//free(irisImg->data);
	//free(irisImg);
	//calculate the origin iris center coordinates on the eye image and return the values of the iris circle
	pIris.x = colIris + setVal.rect.x;//setVal.rect.x, setVal.rect.y为iris矩形框的左上角顶点
	pIris.y = rowIris + setVal.rect.y;
	r_Iris = rIris;

	//cv::circle(mEye, pIris, rIris, Scalar(0, 255, 0), 3, 8, 0);// circle outline
	//cv::imshow("iris", mEye);

	out.pIris = pIris;
	out.r_iris = r_Iris;
	
}