#include "stdafx.h"
//---------------------------------��ͷ�ļ��������ռ�������֡�----------------------------
//		����������������ʹ�õ�ͷ�ļ��������ռ�
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

string video_format = ".avi"; //������Ƶ��ʽ   ȫ�ֱ�������
string img_format = ".jpg"; //����ͼƬ�����ʽ   
string str1 = "_01";  //��һ����Ƶ��׺_01
string str2 = "_02";  //�ڶ�����Ƶ��׺_02
string str3 = "\\";
string str4 = "eye_";   //�۾����󱣴�
string str5 = "iris_right_"; //��Ĥ�������۱���
string str6 = "iris_left_"; //��Ĥ�������۱���

int main(int argc, char* argv[])
{

	//Mat eyetemplate = imread("D:\\֣��2021\\6.�о�����\\01Strabismus\\PolyU\\Codes\\eyetemplate.jpg", 1); //��1������˫��ģ��
	Mat eyetemplate_r = imread("D:\\֣��2021\\6.�о�����\\01Strabismus\\PolyU\\Codes\\eyetemplate_r10.jpg", 1); //��1����������ģ��
	cv::imshow("template", eyetemplate_r);
	waitKey(0);
	return 0;
}
