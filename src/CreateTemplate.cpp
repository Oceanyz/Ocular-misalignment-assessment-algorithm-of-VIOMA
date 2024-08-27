#include "stdafx.h"
#include "CreateTemplate.h"
#include "Masek.h"
//#include "EncodeLee.h"
#include "ImageUtility.h"
#include "FindPupilCircleNew.h"
#include "FindIrisCircle.h"
//#include "FindHighLights.h"
//#include "FindEyelidCurve.h"
//#include "Normalization.h"

#include <iostream>

// Run left and right eye together for video
void CreateTemplate::newCreateIrisTemplate(IplImage* eyeImg,
	              int *center_y, int *center_x, int *radius,
										int dataType)
{
	ImageUtility *imgUtil = NULL;
	/*
	// Load the image as gray scale
	IplImage* eyeImg = NULL;
	eyeImg = cvLoadImage(fileName,0);
	if(eyeImg == NULL)
	{
		cout << "Failed to load the file" << endl;
		return;
	}
	*/
    IplImage* grayImg = NULL;
    grayImg = cvCreateImage(cvSize(eyeImg->width, eyeImg->height), 8, 1);
    cvCopyImage(eyeImg, grayImg);

	if(grayImg == NULL)
	{
		cout << "Failed to load the file" << endl;
		return;
	}		
	
	/********************************************************
	 * Iris Segmentation
	 *********************************************************/	
	//PUPIL INPUTS	
	float nScale = 1.0;		
	const int speed_m = 1;// Default 1
	int alpha = 20; // Alpha value for contrast threshold
	// Setup the parameters to avoid that noise caused by reflections and 
    // eyelashes covers the pupil
	double ratio4Circle = 1.0;
    // Initialize for Closing and Opening process
	int closeItr = 2;//dilate->erode
	int openItr = 3;//erode->dilate
	double norm = 256.0;//

	//IRIS INPUTS
	double scaling = 0.4;// Default
	double lowThres = 0.10;// Default 0.11
	double highThres = 0.15;// Default 0.19
	/*
	if(dataType == NIR_IRIS_STILL) //classical iris image  
	{
		nScale = 2.0;
		alpha = 25;		
		ratio4Circle = 0.90;
		closeItr = 2;
		openItr = 3;
		scaling = 0.4;
	}*/
	if(dataType == NIR_IRIS_STILL) //classical iris image  changed by zhengyang***************************************************************
	{
		nScale = 1.0;  //修改ncscale=1,这个参数取决于图像的尺寸
		alpha = 50;		//alpha为计算自动化阈值的附加常数
		ratio4Circle = 0.90; //ratio4circle为算法在做椭圆拟合时的长短轴之比 因为我们的数据近似圆形，所以ratio4circle设置为0.8
		closeItr = 2;
		openItr = 3;//3
		scaling = 0.6;
	}	
	else if(dataType == NIR_FACE_VIDEO) // Distant video frame
	{	
		nScale = 1.0;
		alpha = 20;		
		ratio4Circle = 0.65;
		closeItr = 0;
		openItr = 3;
		scaling = 0.45;
	}

	/*defined
	ICE2005_IRIS_LG2200		01 (CODE: ICE)
	MBGC_IRIS_LG2200		02 (CODE: MIL)
	MBGC_FACE_IOM			03 (CODE: MFI)	
	ND_IRIS20_LG4000		04 (CODE: N20)
	ND_IRIS48_LGICAM		05 (CODE: N48)
	ND_IRIS49_IRISGUARD		06 (CODE: N49)
	ND_IRIS59_CFAIRS		07 (CODE: N59)*/

	if(dataType == ICE2005_IRIS_LG2200) // Classic still images
	{
		nScale = 2.0;
		alpha = 20;		
		ratio4Circle = 0.92;
		closeItr = 2;
		openItr = 3;
		scaling = 0.25;
	}
	else if(dataType == MBGC_IRIS_LG2200) //02 (CODE: MIL)
	{
		nScale = 2.0;
		alpha = 30;		
		ratio4Circle = 0.92;
		closeItr = 2;
		openItr = 1;
		scaling = 0.2;
	}
	else if(dataType == MBGC_FACE_IOM) // Distant video imagery
	{
		nScale = 1.0;		
		alpha = 20;		
		ratio4Circle = 0.65;
		closeItr = 0;
		openItr = 3;
		scaling = 0.45;		
	}	
	else if(dataType == ND_IRIS20_LG4000) //04 (CODE: N20)
	{
		nScale = 2.0;		
		alpha = 38;		
		ratio4Circle = 0.92;
		closeItr = 0;
		openItr = 4;
		scaling = 0.3;
	}
	else if(dataType == ND_IRIS48_LGICAM) //05 (CODE: N48)
	{
		nScale = 2.0;
		alpha = 40;		
		ratio4Circle = 0.92;
		closeItr = 0;
		openItr = 4;
		scaling = 0.2;
	}
	else if(dataType == ND_IRIS49_IRISGUARD) //06 (CODE: N49)
	{
		nScale = 2.0;
		alpha = 18;		
		ratio4Circle = 0.92;
		closeItr = 3;//best for noScaled Still
		openItr = 2;//best for noScaled Still
		scaling = 0.2;
	}
	else if(dataType == ND_IRIS59_CFAIRS) //07 (CODE: N59)
	{
		nScale = 2.0;
		alpha = 4;		
		ratio4Circle = 0.92;
		closeItr = 0;//best for noScaled Still
		openItr = 3;//best for noScaled Still
		scaling = 0.4;
	}
	//***************************************************修改瞳孔的最大值
	//const int rPupilMax = (int) (42*nScale);// Maximum radius of pupil's circle
	//const int rIrisMax = (int) (82*nScale);// Maximum radius of iris' circle
	
	const int rPupilMax = (int) (45*nScale);// Maximum radius of pupil's circle  yzheng***********************************(7,22)ours (10,22) for casia (20,45) for strabismus videos
	const int rIrisMax = (int) (80*nScale);// Maximum radius of iris' circle	yzheng**********************************(25,45)ours  (28,38)for casia (56,80) for Strabismusvideos
		
		
	//fine the pupil circle using contours
    int pupilCircle[6]={0};

	FindPupilCircleNew::doDetect(grayImg, rPupilMax, ratio4Circle, closeItr, openItr, speed_m, alpha, norm, nScale, pupilCircle);

	CvPoint xyPupil;
	xyPupil.x = pupilCircle[0];
	xyPupil.y = pupilCircle[1];
	int rPupil = pupilCircle[2];
	
	// Draw the pupil circle   *****************************************************************************************************************
	cvCircle(grayImg, cvPoint(xyPupil.x, xyPupil.y), rPupil, CV_RGB(255,255,255), 1, 8);
	ImageUtility::showImage("Pupil Circle", grayImg); 

	
	//ROI for detecting the iris circle
	ImageUtility::SETVALUE setVal = imgUtil->setImage(grayImg, xyPupil, rPupil, rIrisMax, rIrisMax);	//82 is the best for video images, previous 80
	IplImage* setImg = NULL;
	setImg = imgUtil->getROIImage(grayImg, setVal.rect.x, setVal.rect.width, setVal.rect.y, setVal.rect.height);
	if(setImg == NULL)
	{
		cout << "Failed to load the file" << endl;
		return;
	}

	//ImageUtility::showImage("Iris ROI", setImg);  //画出虹膜ROI区域*********************************************************************************

	int centerAdjust=(int)(rIrisMax/4);//(rIrisMax/5); //for video dataset

	//find the iris circle using Hough Transform
    int irisCircle[3]={0};

	FindIrisCircle::doDetect(setImg, rPupil, rIrisMax, scaling, lowThres, highThres, irisCircle);	
	CvPoint xyIris;
	xyIris.x = irisCircle[0];
	xyIris.y = irisCircle[1];	
	int rIris = irisCircle[2];
	
	xyIris = FindIrisCircle::getOriginPoints(xyPupil, xyIris, setVal.p, centerAdjust);
	
	// Draw the iris circle   *****************************************************************************************************************
	//cvCircle(grayImg, cvPoint(xyIris.x, xyIris.y), rIris, CV_RGB(255,255,255), 1, 8);
	//ImageUtility::showImage("Iris Circle", grayImg); 
	*center_x = xyIris.x;
	*center_y = xyIris.y;
	*radius = rIris;
	//*********************************************************************************

	cvReleaseImage(&setImg);
    cvReleaseImage(&grayImg);
	
}

