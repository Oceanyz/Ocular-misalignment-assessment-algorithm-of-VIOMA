#include "stdafx.h"
#include "PupilTracker.h"
#include <iostream>

#include <boost/foreach.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/tbb.h>
#include "delete_curves.h"
#include "GammaCorrection.h"
#include "cvx.h"
using namespace std;
using namespace cv;

namespace 
{
    struct section_guard
    {
        std::string name;
        pupiltracker::tracker_log& log;
        pupiltracker::timer t;
        section_guard(const std::string& name, pupiltracker::tracker_log& log) : name(name), log(log), t() {  }
        ~section_guard() { log.add(name, t); }
        operator bool() const {return false;}
    };

    inline section_guard make_section_guard(const std::string& name, pupiltracker::tracker_log& log)
    {
        return section_guard(name,log);
    }
}

#define SECTION(A,B) if (const section_guard& _section_guard_ = make_section_guard( A , B )) {} else



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

        // Frobenius normalized values
        //
        // Want norm = 1 where norm = sqrt(sum(pixelvals^2)), so:
        //  sqrt(count_inner*val_inner^2 + count_outer*val_outer^2) = 1
        //
        // Also want sum(pixelvals) = 0, so:
        //  count_inner*val_inner + count_outer*val_outer = 0
        //
        // Solving both of these gives:
        //val_inner = std::sqrt( (double)count_outer/(count_inner*count_outer + sq(count_inner)) );
        //val_outer = -std::sqrt( (double)count_inner/(count_inner*count_outer + sq(count_outer)) );

        // Square radius normalised values
        //
        // Want the response to be scale-invariant, so scale it by the number of pixels inside it:
        //  val_inner = 1/count = 1/r_outer^2
        //
        // Also want sum(pixelvals) = 0, so:
        //  count_inner*val_inner + count_outer*val_outer = 0
        //
        // Hence:
        val_inner = 1.0 / (r_inner*r_inner);
        val_outer = -val_inner*count_inner/count_outer;

    }

    double val_inner, val_outer;
    int r_inner, r_outer;
};

cv::RotatedRect fitEllipse(const std::vector<pupiltracker::EdgePoint>& edgePoints)
{
    std::vector<cv::Point2f> points;
    points.reserve(edgePoints.size());

    BOOST_FOREACH(const pupiltracker::EdgePoint& e, edgePoints)
        points.push_back(e.point);

    return cv::fitEllipse(points);
}

void filter_edges(cv::Mat *edge, int start_xx, int end_xx, int start_yy, int end_yy)
{
		int start_x=start_xx+2;
		int end_x=end_xx-2;
		int start_y=start_yy+2;
		int end_y=end_yy-2;

		if(start_x<2) start_x=2;
		if(end_x>edge->cols-2) end_x=edge->cols-2;
		if(start_y<2) start_y=2;
		if(end_y>edge->rows-2) end_y=edge->rows-2;

		for(int j=start_y; j<end_y; j++)
			for(int i=start_x; i<end_x; i++){
				int box[9];

				box[4]=(int)edge->data[(edge->cols*(j))+(i)];

				if(box[4]){
				box[1]=(int)edge->data[(edge->cols*(j-1))+(i)];
				box[3]=(int)edge->data[(edge->cols*(j))+(i-1)];
				box[5]=(int)edge->data[(edge->cols*(j))+(i+1)];
				box[7]=(int)edge->data[(edge->cols*(j+1))+(i)];

				if((box[5] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
				if((box[5] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
				if((box[3] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
				if((box[3] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
				//细化边沿
				}
		}
		//too many neigbours具有超过3个邻居的边沿点被删掉，因为它倾向于连接多个边沿线
		for(int j=start_y; j<end_y; j++)
			for(int i=start_x; i<end_x; i++){
				
				int neig=0;
				for(int k1=-1;k1<2;k1++)
					for(int k2=-1;k2<2;k2++){

						if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
							neig++;
				}

				if(neig>3)
					edge->data[(edge->cols*(j))+(i)]=0;

		}
};

bool pupiltracker::findPupilEllipse(
    const pupiltracker::TrackerParams& params,
    const cv::Mat& m,

    pupiltracker::findPupilEllipse_out& out,
    pupiltracker::tracker_log& log
    )
{
    // --------------------
    // Convert to greyscale
    // --------------------

	cv::Mat_<uchar> srcEye;

    SECTION("Grey and crop", log)
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
	cv::Mat mEye(srcEye.rows, srcEye.cols, srcEye.type());
	if (params.Conduct_Gamma)  //true则执行Haar搜索
	{
		GammaCorrection(srcEye, mEye, 0.5);  //srcEye为原始眼睛图象；mEye为gamma变换后眼睛图象
	}
	else
	{
		mEye = srcEye;
	}
	
	/*
	//Find best Haar rectangle feature   //yzheng***************************************
	cv::Mat_<int> mEyeIntegral1; // 积分图像,为了和原haarSurroundFeature积分图区分，第一个haar积分图为mEyeIntegral1
	SECTION("Integral image1", log)
    {
        cv::integral(mEye, mEyeIntegral1);
    }
	
	 int Length2 = mEye.size().width*3/4;//150 for our dataset 384 for dataset 1, 600 for Swirski  对于眼睛区域图像，眼眶长度约占图像宽度的2/3
	 int wmin = Length2*1/4; //上下眼睑的间隔约是眼睛长度的1/5到1/2
	 int wmax = Length2*1/2; //
	 //int Length2 = 200;//150 for our dataset 384 for dataset 1, 600 for Swirski  对于眼睛区域图像，眼眶长度约占图像宽度的2/3  200,35,70
	 //int wmin = 35; //上下眼睑的间隔约是眼睛长度的1/5到1/2
	 //int wmax = 70; //
	 cv::Mat_<uchar> mHaarshrink;
	 int Haarwidth2 = 0; 
	 cv::Point2f pHaarshrink2;
	 cv::Point2f pHaarshrink;  //第一个Haarfeature返回的特征区域左上角坐标
	 //SECTION("Haar rectangle response", log)
	 if (params.Conduct_HaarRectangle)
	 {
		std::pair<double,cv::Point2f> maxValOut2;
		double maxResponse2 = 0;
		int x_step2=4, y_step2=4,w_step2=5;
		for(int w=wmin; w<=wmax; w+=w_step2) //Swirski
		{
				for(int y=1; y<mEye.rows-2*w+1; y+=y_step2)
					for(int x=1; x<mEye.cols-Length2+1; x+=x_step2)
					{
						int white_zone2 = mEyeIntegral1[y+w][x] + mEyeIntegral1[y+2*w][x+Length2] - mEyeIntegral1[y+w][x+Length2] - mEyeIntegral1[y+2*w][x];
						int black_zone2 = mEyeIntegral1[y][x] + mEyeIntegral1[y+w][x+Length2] - mEyeIntegral1[y][x+Length2] - mEyeIntegral1[y+w][x];
						int response2 = white_zone2 - black_zone2;
						
						if (response2 > maxResponse2)
						{
							maxValOut2.first = response2;
							maxValOut2.second = cv::Point(x,y);

							maxResponse2 = response2;
							maxResponse2 = maxValOut2.first;
							// Set return values
							pHaarshrink2 = maxValOut2.second;
							Haarwidth2 = 2*w;
						}   
					}
		}
		 
		 if (Haarwidth2 == 0)
			 mHaarshrink = mEye;
		 else
		 {
			cv::Rect haarshrink2 = cv::Rect(pHaarshrink2.x, pHaarshrink2.y, Length2, Haarwidth2);	
			mHaarshrink = mEye(haarshrink2);
		 }
		cv::imshow("pupil_step1", mHaarshrink);
		pHaarshrink.x = pHaarshrink2.x;   
		pHaarshrink.y = pHaarshrink2.y;//yzheng **********************************************************************
	 }
	 else  //如果不执行haarrectangle
	 {
		 mHaarshrink = mEye;
		 pHaarshrink.x = 0;
		 pHaarshrink.y = 0;
	 }
	
	
	cv::Mat_<int32_t> mEyeIntegral;
    int padding = 2*params.Radius_Max;//2
	
    SECTION("Integral image", log)
    {
        cv::Mat mEyePad;
        // Need to pad by an additional 1 to get bottom & right edges.
        cv::copyMakeBorder(mHaarshrink, mEyePad, padding, padding, padding, padding, cv::BORDER_REPLICATE);
        cv::integral(mEyePad, mEyeIntegral);
    }

    cv::Point2f pHaarPupil;
    int haarRadius = 0;

	SECTION("Haar surrounded responses", log)
	{
		const int rstep = 2;
		const int ystep = 4;
		const int xstep = 4;

		double minResponse = std::numeric_limits<double>::infinity();

		for (int r = params.Radius_Min; r < params.Radius_Max; r += rstep)
		{
			// Get Haar feature
			int r_inner = r;
			int r_outer = 3 * r;
			HaarSurroundFeature f(r_inner, r_outer);

			// Use TBB for rows
			std::pair<double, cv::Point2f> minRadiusResponse = tbb::parallel_reduce(
				tbb::blocked_range<int>(0, (mHaarshrink.rows - r - r - 1) / ystep + 1, ((mHaarshrink.rows - r - r - 1) / ystep + 1) / 2 > 0 ? ((mHaarshrink.rows - r - r - 1) / ystep + 1) / 2 : 1),
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

					for (int x = r; x < mHaarshrink.cols - r; x += xstep)
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
				pHaarPupil = minRadiusResponse.second;
				haarRadius = r;
			}
		}
	}
	// Paradoxically, a good Haar fit won't catch the entire pupil, so expand it a bit
	haarRadius = (int)(haarRadius * SQRT_2);

	// ---------------------------
	// Pupil ROI around Haar point
	// ---------------------------
	pHaarPupil.x += pHaarshrink.x;
	pHaarPupil.y += pHaarshrink.y;
	cv::Rect roiHaarPupil = cvx::roiAround(cv::Point(pHaarPupil.x, pHaarPupil.y), haarRadius);
	cv::Mat_<uchar> mHaarPupil;//yzheng***************************************************************************************
	cvx::getROI(srcEye, mHaarPupil, roiHaarPupil);
	out.roiHaarPupil = roiHaarPupil;
	out.mHaarPupil = mHaarPupil;
	//cv::imshow("pupil_step2", mHaarPupil);
	//cv::waitKey(0);
	*/
    // --------------------------------------------------
    // Get histogram of pupil region, segment with KMeans
    // --------------------------------------------------

    const int bins = 256;
    cv::Mat_<float> hist;
    SECTION("Histogram", log)
    {
        int channels[] = {0};
        int sizes[] = {bins};
        float range[2] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&mEye, 1, channels, cv::Mat(), hist, 1, sizes, ranges);
    }

    out.histPupil = hist;

    float threshold;
	float Cannythreshold;//yzheng *****************************
	{
        // Try various candidate centres, return the one with minimal label distance
        float candidate0[4] = {13, 20, 30, 50}; //{0, 0}
        float candidate1[4] = {36, 60, 60, 100};//{128, 255}
		float candidate2[4] = {60, 90, 90, 180};
        float bestDist = std::numeric_limits<float>::infinity();
        float bestThreshold = std::numeric_limits<float>::quiet_NaN();
		float bestCannyThreshold = std::numeric_limits<float>::quiet_NaN();//yzheng*********************************************************

        for (int i = 0; i < 4; i++)
        {
            cv::Mat_<uchar> labels;
			//float centres[2] = {candidate0[i], candidate1[i]};
            float centres[3] = {candidate0[i], candidate1[i], candidate2[i]};
            float dist = cvx::histKmeans(hist, 0, 256, 3, centres, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0.0));
			
			std::cout<<"centres "<<centres[0]<<" "<<centres[1]<<" "<<centres[2]<<std::endl;
            //float thisthreshold = (centres[0] + centres[1] )*0.5;  //能把两个类别分开的阈值是两个聚类中心和的一半
			//float thisthreshold = (centres[0] + centres[1] )*0.2;  //能把两个类别分开的阈值是两个聚类中心和的一半
			float thisthreshold = centres[0] * 1.2;//1.2
			float thisCannythreshold = centres[1]-centres[0];//yzheng ***********************

			if (dist < bestDist && !(thisthreshold != thisthreshold))
            {
                bestDist = dist;
                bestThreshold = thisthreshold;
				bestCannyThreshold = thisCannythreshold;//yzheng *****************************
            }
        }
        if ((bestThreshold != bestThreshold))
        {
            // If kmeans gives a degenerate solution, exit early
            return false;
        }
        
        threshold = bestThreshold;
		Cannythreshold = bestCannyThreshold;//yzheng *****************************
		
    }
    cv::Mat_<uchar> mPupilThresh;
    SECTION("Threshold", log)
    {
        cv::threshold(mEye, mPupilThresh, threshold, 255, cv::THRESH_BINARY_INV);
    }

    out.threshold = threshold;
    out.mPupilThresh = mPupilThresh;
	//cv::imshow("pupil_step3", mPupilThresh);
	
	//waitKey(0);
    // ---------------------------------------------
    // Find best region in the segmented pupil image
    // ---------------------------------------------

    cv::Rect bbPupilThresh;
    cv::RotatedRect elPupilThresh;

    SECTION("Find best region", log)
    {
        cv::Mat_<uchar> mPupilContours = mPupilThresh.clone();
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mPupilContours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        if (contours.size() == 0)
            return false;

        std::vector<cv::Point>& maxContour = contours[0];
        double maxContourArea = cv::contourArea(maxContour);
        BOOST_FOREACH(std::vector<cv::Point>& c, contours)
        {
            double area = cv::contourArea(c);
            if (area > maxContourArea)
            {
                maxContourArea = area;
                maxContour = c;
            }
        }

        cv::Moments momentsPupilThresh = cv::moments(maxContour);

        bbPupilThresh = cv::boundingRect(maxContour);
        elPupilThresh = cvx::fitEllipse(momentsPupilThresh);
		
        // Shift best region into eye coords (instead of pupil region coords), and get ROI
       // bbPupilThresh.x += roiHaarPupil.x;
        //bbPupilThresh.y += roiHaarPupil.y;
        //elPupilThresh.center.x += roiHaarPupil.x;
        //elPupilThresh.center.y += roiHaarPupil.y;
    }

    out.bbPupilThresh = bbPupilThresh;
    out.elPupilThresh = elPupilThresh;
	
	
    // ------------------------------
    // Find edges in new pupil region
    // ------------------------------
	
	//int rough_radius = 1.0*max(bbPupilThresh.width,bbPupilThresh.height)*0.5;//bbPupilThresh瞳孔区域的粗略半径yzheng**************************
	int rough_radius = SQRT_2*max(bbPupilThresh.width,bbPupilThresh.height)*0.5;//bbPupilThresh瞳孔区域的粗略半径yzheng**************************
	//int rough_radius = SQRT_2*max(elPupilThresh.size.width, elPupilThresh.size.height)*0.5; //elPupilThresh瞳孔区域的粗略半径 yzheng**************************
	float Aspectratio = 1.1f*max(bbPupilThresh.width,bbPupilThresh.height)/min(bbPupilThresh.width,bbPupilThresh.height);
    cv::Mat_<uchar> mPupil, mPupilOpened, mPupilBlurred, mPupilEdges, mPupilEdges1, mPupilEdges2, mPupilEdges3;
    cv::Mat_<float> mPupilSobelX, mPupilSobelY;
    cv::Rect bbPupil;
    //cv::Rect roiPupil = cvx::roiAround(cv::Point(elPupilThresh.center.x, elPupilThresh.center.y), haarRadius);
	cv::Rect roiPupil = cvx::roiAround(cv::Point(elPupilThresh.center.x, elPupilThresh.center.y), rough_radius); //yzheng**********************************
    SECTION("Pupil preprocessing", log)
    {
        const int padding = 3;

        cv::Rect roiPadded(roiPupil.x-padding, roiPupil.y-padding, roiPupil.width+2*padding, roiPupil.height+2*padding);
        // First get an ROI around the approximate pupil location
        cvx::getROI(mEye, mPupil, roiPadded, cv::BORDER_REPLICATE);
		
		/*
        cv::Mat morphologyDisk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mPupil, mPupilOpened, cv::MORPH_OPEN, morphologyDisk, cv::Point(-1,-1), 2);

        if (params.CannyBlur > 0)
        {
            cv::GaussianBlur(mPupilOpened, mPupilBlurred, cv::Size(), params.CannyBlur);
        }
        else
        {
            mPupilBlurred = mPupilOpened;
        }

        cv::Sobel(mPupilBlurred, mPupilSobelX, CV_32F, 1, 0, 3);
        cv::Sobel(mPupilBlurred, mPupilSobelY, CV_32F, 0, 1, 3);

        cv::Canny(mPupilBlurred, mPupilEdges, params.CannyThreshold1, params.CannyThreshold2);*/

		//yzheng**************************************************************************************************
		if (params.CannyBlur > 0)         //yzheng*************************************************
        {
            cv::GaussianBlur(mPupil, mPupilBlurred, cv::Size(), params.CannyBlur);
        }
        else
        {
            mPupilBlurred = mPupil;
        }
		
		cv::Mat morphologyDisk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));//cv::MORPH_ELLIPSE,cv::MORPH_RECT,cv::MORPH_CROSS
        cv::morphologyEx(mPupilBlurred, mPupilOpened, cv::MORPH_OPEN, morphologyDisk, cv::Point(-1,-1), 2); //形态学开运算
		
        cv::Sobel(mPupilOpened, mPupilSobelX, CV_32F, 1, 0, 3);
        cv::Sobel(mPupilOpened, mPupilSobelY, CV_32F, 0, 1, 3);

        cv::Canny(mPupilBlurred, mPupilEdges1, Cannythreshold*0.75, Cannythreshold*1.2); //yzheng**********************************************
		//cv::Canny(mPupilBlurred, mPupilEdges1, Cannythreshold*1, Cannythreshold*1.5); //yzheng**********************************************
		
        cv::Rect roiUnpadded(padding,padding,roiPupil.width,roiPupil.height);
        mPupil = cv::Mat(mPupil, roiUnpadded);
        mPupilOpened = cv::Mat(mPupilOpened, roiUnpadded);
        mPupilBlurred = cv::Mat(mPupilBlurred, roiUnpadded);
        mPupilSobelX = cv::Mat(mPupilSobelX, roiUnpadded);
        mPupilSobelY = cv::Mat(mPupilSobelY, roiUnpadded);
        mPupilEdges2 = cv::Mat(mPupilEdges1, roiUnpadded);//yzheng******************************************************
        bbPupil = cvx::boundingBox(mPupil);
    }
	//cv::imshow("pupil_step4", mPupil);
	
	//cv::imshow("canny", mPupilEdges2);
	cv::Mat morphologyDisk2 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2));//对边沿图像执行膨胀操作yzheng**************************************
	cv::morphologyEx(mPupilEdges2, mPupilEdges3, cv::MORPH_DILATE, morphologyDisk2, cv::Point(-1,-1), 1);//yzheng***********************************
	//imshow("edge3",mPupilEdges3);
	//waitKey(1000);
	//mPupilEdges = mPupilEdges2;
	mPupilEdges = delete_curves(&mPupilEdges3, 3);  //yzheng******************************************************执行边沿图像滤波，消除孤立点和直线
	//cv::imshow("edge_step1",mPupilEdges);
	filter_edges(&mPupilEdges, 0, mPupilEdges.cols , 0, mPupilEdges.rows);//yzheng ********************************************
	//cv::imshow("edge_step2",mPupilEdges);
	//cv::waitKey(0);
	
	//waitKey(1000);

    out.roiPupil = roiPupil;
    out.mPupil = mPupil;
	
    //out.mPupilOpened = mPupilOpened;
    //out.mPupilBlurred = mPupilBlurred;
    out.mPupilSobelX = mPupilSobelX;
    out.mPupilSobelY = mPupilSobelY;
    out.mPupilEdges = mPupilEdges;
	out.mPupilBlurred = mPupilEdges2;

    // -----------------------------------------------
    // Get points on edges, optionally using starburst
    // -----------------------------------------------

    std::vector<cv::Point2f> edgePoints;

    if (params.StarburstPoints > 0)
    {
        SECTION("Starburst", log)
        {
            // Starburst from initial pupil approximation, stopping when an edge is hit.
            // Collect all edge points into a vector

            // The initial pupil approximations are:
            //    Centre of mass of thresholded region
            //    Halfway along the major axis (calculated form second moments) in each direction

            tbb::concurrent_vector<cv::Point2f> edgePointsConcurrent;

            cv::Vec2f elPupil_majorAxis = cvx::majorAxis(elPupilThresh);
            std::vector<cv::Point2f> centres;
            centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y));
            centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y) + cv::Point2f(elPupil_majorAxis));
            centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y) - cv::Point2f(elPupil_majorAxis));

            BOOST_FOREACH(const cv::Point2f& centre, centres) {
                tbb::parallel_for(0, params.StarburstPoints, [&] (int i) {
                    double theta = i * 2*PI/params.StarburstPoints;

                    // Initialise centre and direction vector
                    cv::Point2f pDir((float)std::cos(theta), (float)std::sin(theta));  

                    int t = 1;
                    cv::Point p = centre + (t * pDir);
                    while(p.inside(bbPupil))
                    {
                        uchar val = mPupilEdges(p);

                        if (val > 0)
                        {
                            float dx = mPupilSobelX(p);
                            float dy = mPupilSobelY(p);

                            float cdirx = p.x - (elPupilThresh.center.x - roiPupil.x);
                            float cdiry = p.y - (elPupilThresh.center.y - roiPupil.y);

                            // Check edge direction
                            double dirCheck = dx*cdirx + dy*cdiry;

                            if (dirCheck > 0)
                            {
                                // We've hit an edge
                                edgePointsConcurrent.push_back(cv::Point2f(p.x + 0.5f, p.y + 0.5f));
                                break;
                            }
                        }

                        ++t;
                        p = centre + (t * pDir);
                    }
                });
            }

            edgePoints = std::vector<cv::Point2f>(edgePointsConcurrent.begin(), edgePointsConcurrent.end());


            // Remove duplicate edge points
            std::sort(edgePoints.begin(), edgePoints.end(), [] (const cv::Point2f& p1, const cv::Point2f& p2) -> bool {
                if (p1.x == p2.x)
                    return p1.y < p2.y;
                else
                    return p1.x < p2.x;
            });
            edgePoints.erase( std::unique( edgePoints.begin(), edgePoints.end() ), edgePoints.end() );

            if (edgePoints.size() < params.StarburstPoints/2)
                return false;
        }
    }
    else
    {
        SECTION("Non-zero value finder", log)
        {
            for(int y = 0; y < mPupilEdges.rows; y++)
            {
                uchar* val = mPupilEdges[y];
                for(int x = 0; x < mPupilEdges.cols; x++, val++)
                {
                    if(*val == 0)
                        continue;

                    edgePoints.push_back(cv::Point2f(x + 0.5f, y + 0.5f));
                }
            }
        }
    }


    // ---------------------------
    // Fit an ellipse to the edges
    // ---------------------------
	 // Use TBB for RANSAC
	struct EllipseRansac_out {
		std::vector<cv::Point2f> bestInliers;
		cv::RotatedRect bestEllipse;
		double bestEllipseGoodness;
		int earlyRejections;
		bool earlyTermination;

		EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
	};
	struct EllipseRansac {
		const TrackerParams& params;
		const std::vector<cv::Point2f>& edgePoints;
		int n;
		const cv::Rect& bb;
		const cv::Mat_<float>& mDX;
		const cv::Mat_<float>& mDY;
		cv::Mat_<uchar>& mEdges; //yzheng
		float Aspectratio;
		int earlyRejections;
		bool earlyTermination;

		EllipseRansac_out out;

		EllipseRansac(
			const TrackerParams& params,
			const std::vector<cv::Point2f>& edgePoints,
			int n,
			const cv::Rect& bb,
			const cv::Mat_<float>& mDX,
			const cv::Mat_<float>& mDY,
			cv::Mat_<uchar>& mEdges,
			float Aspectratio)
			: params(params), edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), mEdges(mEdges), Aspectratio(Aspectratio), earlyTermination(false), earlyRejections(0)
		{
		}

		EllipseRansac(EllipseRansac& other, tbb::split)
			: params(other.params), edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), mEdges(other.mEdges), Aspectratio(other.Aspectratio), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections)
		{
			//std::cout << "Ransac split" << std::endl;
		}

		void operator()(const tbb::blocked_range<size_t>& r)
		{
			//yzheng************************************************************************************************************************************
			std::vector<cv::Point2f> sector1,sector2,sector3,sector4,sector5,sector6,sector7,sector8;
			std::vector<std::vector<cv::Point2f>> sector;
			Point center = Point(bb.width/2,bb.height/2);//在提取的边沿图像上，预测瞳孔中心为（0.5*width，0.5*height）
			for (int i = 1;i<edgePoints.size();i++)
			{					

					if ( (edgePoints.at(i).x >= center.x) & (edgePoints.at(i).y >= center.y) ){
						if ( (edgePoints.at(i).y-center.y) <= (edgePoints.at(i).x-center.x) ) {
							sector1.push_back(edgePoints.at(i));}
						else{
							sector2.push_back(edgePoints.at(i));}
					}
					if ( (edgePoints.at(i).x < center.x) & (edgePoints.at(i).y > center.y) ){
						if ( (edgePoints.at(i).y-center.y) <= abs(edgePoints.at(i).x-center.x) ) {
							sector4.push_back(edgePoints.at(i));}
						else{
							sector3.push_back(edgePoints.at(i));}
					}
					if ( (edgePoints.at(i).x <= center.x) & (edgePoints.at(i).y <= center.y) ){
						if ( abs(edgePoints.at(i).y-center.y) <= abs(edgePoints.at(i).x-center.x) ) {
							sector5.push_back(edgePoints.at(i));}
						else{
							sector6.push_back(edgePoints.at(i));}
					}
					if ( (edgePoints.at(i).x > center.x) & (edgePoints.at(i).y < center.y) ){
						if ( abs(edgePoints.at(i).y-center.y) <= abs(edgePoints.at(i).x-center.x) ) {
							sector8.push_back(edgePoints.at(i));}
						else{
							sector7.push_back(edgePoints.at(i));}
					}
			}
			sector.push_back(sector1);
			sector.push_back(sector2);
			sector.push_back(sector3);
			sector.push_back(sector4);
			sector.push_back(sector5);
			sector.push_back(sector6);
			sector.push_back(sector7);
			sector.push_back(sector8);
			//yzheng*****************************************************************************************************************
			if (out.earlyTermination)
				return;
			//int iterations = 0; //yzheng*************************************************
			//std::cout << "Ransac start (" << (r.end()-r.begin()) << " elements)" << std::endl;
			for( size_t i=r.begin(); i!=r.end(); ++i )
			{
				// Ransac Iteration
				//iterations++; //yzheng*********************************************
				//std::cout<<"iterations: "<<iterations<<std::endl;//yzheng*****************************************************

				std::vector<cv::Point2f> sample;
				//if (params.Seed >= 0)
				//    sample = randomSubset(edgePoints, n, static_cast<unsigned int>(i + params.Seed));
				//else
				//    sample = randomSubset(edgePoints, n);
				//从8个扇区里面各随机选择一个点组成sample yzheng*********************************************
						
				while(sample.size()<n)  //产生sample，先随机选择扇区，再随机选择扇区里的点
				{
						size_t j = random(0, sector.size()-1);//先随机选择一个扇区
						if (sector.at(j).size()>0) {
							size_t idx = random(0, sector.at(j).size()-1);
							sample.push_back(sector.at(j).at(idx));
						}
				}//yzheng**********************************************************************************************
						
				cv::RotatedRect ellipseSampleFit = fitEllipse(sample);
					
				// Normalise ellipse to have width as the major axis.
				if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
				{
					ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
					std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
				}

				cv::Size s = ellipseSampleFit.size;
				/*// Discard useless ellipses early
				if (!ellipseSampleFit.center.inside(bb)
					|| s.height > params.Radius_Max*2 //changed by yzheng
					|| s.width > params.Radius_Max*2
					|| s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
					|| s.height > 4*s.width
					|| s.width > 4*s.height
					)
				{
					// Bad ellipse! Go to your room!
					continue;
				}
				*/
				if (!ellipseSampleFit.center.inside(bb)
					||  1.0f*s.width/s.height > Aspectratio //changed by yzheng
					|| s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
					|| s.width > params.Radius_Max*2
					)
				{
					// Bad ellipse! Go to your room!
					continue;
				}

				// Use conic section's algebraic distance as an error measure
				ConicSection conicSampleFit(ellipseSampleFit);
						
				// Check if sample's gradients are correctly oriented
				if (params.EarlyRejection)
				{
					bool gradientCorrect = true;
					BOOST_FOREACH(const cv::Point2f& p, sample)
					{
						cv::Point2f grad = conicSampleFit.algebraicGradientDir(p);
						float dx = mDX(cv::Point(p.x, p.y));
						float dy = mDY(cv::Point(p.x, p.y));

						float dotProd = dx*grad.x + dy*grad.y;

						gradientCorrect &= dotProd > 0;
					}
					if (!gradientCorrect)
					{
						out.earlyRejections++;
						continue;
					}
				}
						
				// Assume that the sample is the only inliers

				cv::RotatedRect ellipseInlierFit = ellipseSampleFit;
				ConicSection conicInlierFit = conicSampleFit;
				std::vector<cv::Point2f> inliers, prevInliers;
				double dist = 0;//yzheng*************************************************
				// Iteratively find inliers, and re-fit the ellipse
				for (int i = 0; i < params.InlierIterations; ++i)
				{
							
					// Get error scale for 1px out on the minor axis
					cv::Point2f minorAxis(-std::sin(PI/180.0*ellipseInlierFit.angle), std::cos(PI/180.0*ellipseInlierFit.angle));
					cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height/2 + 1)*minorAxis;
					float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
					float errorScale = 1.0f/errOf1px;
					/*
					//generate a contourof an ellipse from bounding box of the ellipse
					vector<Point2f> contour;
					int dev = 150;
					float dtheta = acos(-1)*2/dev;
					float theta = 0;
					for (int r = 0; r < dev; ++r, theta+=dtheta) {
						cv::Point2f p(
								ellipseInlierFit.center.x + ellipseInlierFit.size.width/2*cos(theta)*cos(ellipseInlierFit.angle) + ellipseInlierFit.size.height/2*sin(theta)*sin(ellipseInlierFit.angle),
								ellipseInlierFit.center.y - ellipseInlierFit.size.width/2*cos(theta)*sin(ellipseInlierFit.angle) + ellipseInlierFit.size.height/2*sin(theta)*cos(ellipseInlierFit.angle));
						contour.push_back(p);
					}*/
					// Find inliers
					inliers.reserve(edgePoints.size());
					inliers.clear();//yzheng*****************************************
					double dist = 0;//yzheng*******************************************
					const float MAX_ERR = 2;
					/*
					float err;
					BOOST_FOREACH(const cv::Point2f& p, edgePoints)
					{
						err = abs(pointPolygonTest(contour,p, true));
						dist += err;//yzheng************************************************err < MAX_ERR
						if (err < MAX_ERR){
							inliers.push_back(p);}
							
					}*/
							
					BOOST_FOREACH(const cv::Point2f& p, edgePoints)
					{
						float err = errorScale*conicInlierFit.distance(p);

						if (err*err < MAX_ERR*MAX_ERR){
							inliers.push_back(p);}
							dist += abs(err);//yzheng************************************************
					}
							
					//cout<<"***** "<<inliers.size()<<endl;
					if (inliers.size() < n) {
						inliers.clear();
						continue;
					}

					// Refit ellipse to inliers
					ellipseInlierFit = fitEllipse(inliers);
					conicInlierFit = ConicSection(ellipseInlierFit);

					// Normalise ellipse to have width as the major axis.
					if (ellipseInlierFit.size.height > ellipseInlierFit.size.width)
					{
						ellipseInlierFit.angle = std::fmod(ellipseInlierFit.angle + 90, 180);
						std::swap(ellipseInlierFit.size.height, ellipseInlierFit.size.width);
					}
				}
				if (inliers.empty())
					continue;

				// Discard useless ellipses again
				s = ellipseInlierFit.size;
				/*if (!ellipseInlierFit.center.inside(bb)
					|| s.height > params.Radius_Max*2 //changed by yzheng
					|| s.width > params.Radius_Max*2
					|| s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
					|| s.height > 4*s.width
					|| s.width > 4*s.height
					)
				{
					// Bad ellipse! Go to your room!
					continue;
				}*/
				if (!ellipseInlierFit.center.inside(bb)
					||  1.0f*s.width/s.height > Aspectratio //changed by yzheng
					|| s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
					|| s.width > params.Radius_Max*2
					
					)
				{
					// Bad ellipse! Go to your room!
					continue;
				}		
				// Calculate ellipse goodness
				double ellipseGoodness = 0;
				if (params.ImageAwareSupport)
				{
					BOOST_FOREACH(cv::Point2f& p, inliers)
					{
						cv::Point2f grad = conicInlierFit.algebraicGradientDir(p);
						float dx = mDX(p);
						float dy = mDY(p);

						double edgeStrength = dx*grad.x + dy*grad.y;

						ellipseGoodness += edgeStrength;
					}
				}
				else
				{
					ellipseGoodness = inliers.size();
					//ellipseGoodness = -dist;
				}
						
						
				if (ellipseGoodness > out.bestEllipseGoodness)
				{
					std::swap(out.bestEllipseGoodness, ellipseGoodness);
					std::swap(out.bestInliers, inliers);
					std::swap(out.bestEllipse, ellipseInlierFit);
					//float w = (1.0*out.bestInliers.size())/edgePoints.size(); //yzheng******************************************************
					//if (params.EarlyTerminationPercentage > 0 && w > params.EarlyTerminationPercentage*1.0f/100) //yzheng*********************************** 与Swirski原判断条件一致
					// Early termination, if 90% of points match
					if (params.EarlyTerminationPercentage > 0 && out.bestInliers.size() > params.EarlyTerminationPercentage*edgePoints.size()/100)
							
					{
						earlyTermination = true;
						break;
					}
				}

			}
			//std::cout << "Ransac end" << std::endl;
		}

		void join(EllipseRansac& other)
		{
			//std::cout << "Ransac join" << std::endl;
			if (other.out.bestEllipseGoodness > out.bestEllipseGoodness)
			{
				std::swap(out.bestEllipseGoodness, other.out.bestEllipseGoodness);
				std::swap(out.bestInliers, other.out.bestInliers);
				std::swap(out.bestEllipse, other.out.bestEllipse);
			}
			out.earlyRejections += other.out.earlyRejections;
			earlyTermination |= other.earlyTermination;

			out.earlyTermination = earlyTermination;
		}
	};

    cv::RotatedRect elPupil;
    std::vector<cv::Point2f> inliers;
    SECTION("Ellipse fitting", log)
    {
        // Desired probability that only inliers are selected
        const double p = 0.999;
        // Probability that a point is an inlier
        double w = params.PercentageInliers/100.0;
        // Number of points needed for a model
        const int n = 5;

        if (params.PercentageInliers == 0)
            return false;

        if (edgePoints.size() >= n) // Minimum points for ellipse
        {
            // RANSAC!!!

            double wToN = std::pow(w,n);
            int k = static_cast<int>(std::log(1-p)/std::log(1 - wToN)  + 2*std::sqrt(1 - wToN)/wToN);

            out.ransacIterations = k;

            log.add("k", k);

            //size_t threshold_inlierCount = std::max<size_t>(n, static_cast<size_t>(out.edgePoints.size() * 0.7));

			EllipseRansac ransac(params, edgePoints, n, bbPupil, out.mPupilSobelX, out.mPupilSobelY, out.mPupilEdges, Aspectratio);
            try
            { 
                tbb::parallel_reduce(tbb::blocked_range<size_t>(0,k,k/8), ransac);
            }
            catch (std::exception& e)
            {
                const char* c = e.what();
                std::cerr << e.what() << std::endl;
            }
            inliers = ransac.out.bestInliers;
            log.add("goodness", ransac.out.bestEllipseGoodness);

            out.earlyRejections = ransac.out.earlyRejections;
            out.earlyTermination = ransac.out.earlyTermination;


            cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
            ConicSection conicBestFit(ellipseBestFit);
            BOOST_FOREACH(const cv::Point2f& p, edgePoints)
            {
                cv::Point2f grad = conicBestFit.algebraicGradientDir(p);
                float dx = out.mPupilSobelX(p);
                float dy = out.mPupilSobelY(p);

                out.edgePoints.push_back(EdgePoint(p, dx*grad.x + dy*grad.y));
            }

            elPupil = ellipseBestFit;
            elPupil.center.x += roiPupil.x;
            elPupil.center.y += roiPupil.y;
        }

        if (inliers.size() == 0)
            return false;

        cv::Point2f pPupil = elPupil.center;

        out.pPupil = pPupil;
        out.elPupil = elPupil;
        out.inliers = inliers;

		return true;
    }
	
    return false;
}
