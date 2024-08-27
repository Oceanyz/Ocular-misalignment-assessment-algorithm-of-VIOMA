#include "stdafx.h"
//---------------------------------【头文件、命名空间包含部分】----------------------------
//		描述：包含程序所使用的头文件和命名空间
//------------------------------------------------------------------------------------------------
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <process.h>
#include <fstream>
#include "PupilTracker.h"
#include "IrisTracker.h"
#include "cvx.h"
#include <time.h>
#include <string>
#include <algorithm>
#include "Masek.h"
#include "ImageUtility.h"
#include <unordered_map>
#include "CreateTemplate.h"


using namespace cv;
using namespace std;

string video_format = ".avi"; //定义视频格式   全局变量声明
string img_format = ".jpg"; //定义图片保存格式   
string str1 = "_01";  //第一段视频后缀_01
string str2 = "_02";  //第二段视频后缀_02
string str3 = "\\";
string str4 = "eye_";   //眼睛检测后保存
string str5 = "iris_right_"; //虹膜检测后右眼保存
string str6 = "iris_left_"; //虹膜检测后右眼保存

int main(int argc, char* argv[])
{

	//Mat eyetemplate = imread("D:\\郑洋2021\\6.研究方向\\01Strabismus\\PolyU\\Codes\\eyetemplate.jpg", 1); //【1】载入双眼模板
	Mat eyetemplate_r = imread("D:\\郑洋2021\\6.研究方向\\01Strabismus\\PolyU\\Codes\\eyetemplate_r10.jpg", 1); //【1】载入右眼模板
	cv::imshow("template", eyetemplate_r);
	waitKey(0);
	return 0;
}
