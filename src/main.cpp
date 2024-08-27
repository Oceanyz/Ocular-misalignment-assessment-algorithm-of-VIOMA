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
//#include <unordered_map>
#include "strabismus.h"


using namespace cv;
using namespace std;

//#define WINDOW_NAME1 "��ԭʼͼƬ��"        //Ϊ���ڱ��ⶨ��ĺ� 
//#define WINDOW_NAME2 "��ƥ�䴰�ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 
#define numframes 10
#define Video_Name "CT060"         //Ԥ������Ƶ���֣��޸Ĵ˴���������
#define Video_Dir "D:\\yZheng2021\\6Projects\\01Strabismus\\PolyU\\PolyU_videos"
#define Result_Dir1 "D:\\yZheng2021\\6Projects\\01Strabismus\\PolyU\\Detection_Results\\circles1"
#define Result_Dir2 "D:\\yZheng2021\\6Projects\\01Strabismus\\PolyU\\Detection_Results\\circles2"
string video_format = ".avi"; //������Ƶ��ʽ   ȫ�ֱ�������
string img_format = ".jpg"; //����ͼƬ�����ʽ   
string str1 = "_01";  //��һ����Ƶ��׺_01
string str2 = "_02";  //�ڶ�����Ƶ��׺_02
string str3 = "\\";
string str4 = "eye_";   //�۾����󱣴�
string str5 = "iris_right_"; //��Ĥ�������۱���
string str6 = "iris_left_"; //��Ĥ�������۱���
string pupil_prefix_01 = Result_Dir1 + str3 + Video_Name + str1 + str3 + Video_Name;  //ͫ�׼��������·��ǰ׺ ��Ƶ01
string pupil_prefix_02 = Result_Dir2 + str3 + Video_Name + str2 + str3 + Video_Name;  //ͫ�׼��������·��ǰ׺ ��Ƶ02
string str8 = "pupilcenter_";
string name_pupilresults = str8 + Video_Name + ".txt";
FILE* pupil_center = fopen(name_pupilresults.c_str(), "w");//��ͫ�����ļ�������浽pupilcenter.txt�ļ���
bool Conduct_keyframe_search = FALSE;   //���ڴ�����������ؼ�֡λ�û���һ�£�����ֻ��Ҫ����һ�ιؼ�֡���  FALSE
cv::Mat frame, Eye_right, Eye_left, Eye; //��Ƶ֡������ͼ������ͼ��, �۾�ͼ��
Rect pos_right, pos_left;
int radius_right, radius_left;//��Ĥ���뾶
cv::Point irisright_center, irisleft_center; //��Ĥ�����������
int for_ward = 6;  //ͫ�׼���ڹؼ�֡����ǰ����֡����Ƶ���紦 10  6 44 10 40 
int back_ward = 44; //40

//���ȶ���ͫ�׼��Ĳ���
pupiltracker::findPupilEllipse_out out_right, out_left, out;
pupiltracker::TrackerParams params;
pupiltracker::tracker_log log_right, log_left, log_pupil;
//ͫ�׼������������

int randrange(int low, int high)   /* generates a random number within given range*/
{
	return rand() % (high - low + 1) + low;
}
//-----------------------------------��on_Matching( )������--------------------------------
//          �������ص�����
//-------------------------------------------------------------------------------------------
cv::Point on_Matching(Mat g_srcImage, Mat g_templateImage, int g_nMatchMethod)
{
	//��1�����ֲ�������ʼ��
	Mat srcImage; Mat g_resultImage;
	g_srcImage.copyTo(srcImage);

	//��2����ʼ�����ڽ������ľ���
	int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
	int resultImage_cols = g_srcImage.cols - g_templateImage.cols + 1;
	g_resultImage.create(resultImage_rows, resultImage_cols, CV_32FC1);

	//��3������ƥ��ͱ�׼��
	matchTemplate(g_srcImage, g_templateImage, g_resultImage, g_nMatchMethod);
	normalize(g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat());

	//��4��ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minValue; double maxValue; Point minLocation; Point maxLocation;
	Point matchLocation;
	minMaxLoc(g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());

	//��5�����ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ���Ÿ��ߵ�ƥ����. ������ķ���, ��ֵԽ��ƥ��Ч��Խ��
	//�˾�����OpenCV2��Ϊ��
	//if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
	//�˾�����OpenCV3��Ϊ��
	if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
	}
	else
	{
		matchLocation = maxLocation;
	}

	//��6�����Ƴ����Σ�����ʾ���ս��
	//rectangle(srcImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);
	//rectangle(g_resultImage, matchLocation, Point(matchLocation.x + g_templateImage.cols, matchLocation.y + g_templateImage.rows), Scalar(0, 0, 255), 2, 8, 0);

	return matchLocation;
}
struct compare_xy {
	bool operator ()(cv::Point left, cv::Point right) const {
		return (left.x == right.x ? left.y < right.y : left.x < right.x);
	}
};


//keyframe_search�ļ��˼����ģ��ƥ��ͼ���ϼ��λ�û������ֲ�����߶Σ�����λ���ƶ�������ˮƽ�߶ε���ʼ�ͽ���λ�ã����������ø���Ȥ�ļ���
vector<int>  keyframe_search(vector< vector<int> > tm_results, int th)
{
	int ref_point = 0;  //�������ˮƽ�߶εĲο���
	int delta_x;
	vector<vector<int> > segments;//segments�洢������Ƭ�ε���ʼ�ͽ���λ��
	vector<int> interval(2,0);//Ƭ�ε���ʼ�ͽ���λ��
	vector<int> keyframe(12);
	for (int i = ref_point; i < tm_results.size()-1; i++)
	{
		delta_x = abs(tm_results.at(i + 1).at(1) - tm_results.at(ref_point).at(1)); //��ÿһ֡���ο�֡delta_x
		if (delta_x > th)//С��th˵���Ͳο�����һ��ˮƽ��
		{
			if (i + 1 - ref_point > 90)  //��ס�۾�����СƬ��Ϊ2s��С��2s��ֱ�߶�����
			{
				segments.push_back(interval);
				segments.back().at(0) = ref_point;//ref_pointΪƬ�ε���ʼλ��
				segments.back().at(1) = i; //iΪƬ�ε���ֹλ��
			}
			ref_point = i + 1;
		}
		
	}
	for (int j = 0; j < keyframe.size(); j++)
	{
		keyframe[j] = segments.at(j).at(0);
	}
	
	return keyframe;

}
/*
//pupil_detection��������Ƶ�ؼ�֡���������������Ͻ���ͫ�׼�Ⲣ�������ݺ�ͼ��
vector<vector<float>> pupil_detection(cv::VideoCapture cap, int keyframe, cv::Rect pos, cv::Point center_iris, int radius_iris, std::string pupil_prefix, std::string whicheye)
{
	vector< vector<float> > pupil_results;
	vector<float> pupil(3, 0);
	double position = keyframe - for_ward;
	cap.set(CV_CAP_PROP_POS_FRAMES, position);

	for (unsigned int k = position; k < keyframe + back_ward; ++k)
	{
		cap.read(frame);
		Eye = frame(pos);//���ۻ���������
		
		cv::Rect roi_iris((center_iris.x - radius_iris >= 0) ? center_iris.x - radius_iris: 0, (center_iris.y - radius_iris >= 0) ? center_iris.y - radius_iris : 0, (center_iris.x + radius_iris >= pos.width) ? pos.width - center_iris.x + radius_iris : 2 * radius_iris, (center_iris.y + radius_iris >= pos.height) ? pos.height - center_iris.y + radius_iris : 2 * radius_iris);
		cv::Mat ROI_iris = Eye(roi_iris);
		pupiltracker::findPupilEllipse(params, ROI_iris, out, log_pupil);//�ڸ���Ȥ���۾������Ͻ���ͫ�׼��
		cv::ellipse(ROI_iris, out.elPupil, pupiltracker::cvx::rgb(255, 0, 0), 3);//������Բ����
		cv::circle(ROI_iris, out.pPupil, 3, Scalar(0, 0, 255), -1, 8); //����ͫ������

		pupil_results.push_back(pupil);
		pupil_results.back().at(0) = k;
		pupil_results.back().at(1) = out.pPupil.x; 
		pupil_results.back().at(2) = out.pPupil.y;
		
		fprintf(pupil_center, "%d", k);
		fprintf(pupil_center, "\t%f", out.pPupil.x);
		fprintf(pupil_center, "\t%f\n", out.pPupil.y);
		
		//����ͫ�׼�����ı�����
		std::string postfix = to_string(k);
		if (postfix.size() < 4) postfix = "0" + postfix;//����Ϊ4λ��ʽ ��0824
		string pupildetection = pupil_prefix + whicheye + postfix + ".jpg"; //���ڵ�һ����Ƶpupil_prefixΪpupil_prefix_01,�ڶ���Ϊpupil_prefix_02
		
		cv::imwrite(pupildetection, ROI_iris);
		//imshow("namedWindow", ROI_iris);
		//waitKey(1000);
		
	}
	return pupil_results;
}
*/
vector<vector<float>> pupil_detection(cv::VideoCapture cap, int keyframe, cv::Rect pos, cv::Rect roi_iris, std::string pupil_prefix, std::string whicheye)
{
	vector< vector<float> > pupil_results;
	vector<float> pupil(3, 0);
	double position = keyframe - for_ward;	
	cap.set(CV_CAP_PROP_POS_FRAMES, position);

	for (unsigned int k = position; k < keyframe + back_ward; ++k)	
	{
		cap.read(frame);
		Eye = frame(pos);//���ۻ���������

		//cv::Rect roi_iris((center_iris.x - radius_iris >= 0) ? center_iris.x - radius_iris : 0, (center_iris.y - radius_iris >= 0) ? center_iris.y - radius_iris : 0, (center_iris.x + radius_iris >= pos.width) ? pos.width - center_iris.x + radius_iris : 2 * radius_iris, (center_iris.y + radius_iris >= pos.height) ? pos.height - center_iris.y + radius_iris : 2 * radius_iris);
		cv::Mat ROI_iris = Eye(roi_iris);
		pupiltracker::findPupilEllipse(params, ROI_iris, out, log_pupil);//�ڸ���Ȥ���۾������Ͻ���ͫ�׼��

		cv::RotatedRect elPupil = RotatedRect(Point2f(out.elPupil.center.x + roi_iris.x, out.elPupil.center.y + roi_iris.y), out.elPupil.size, out.elPupil.angle);
		cv::ellipse(Eye, elPupil, pupiltracker::cvx::rgb(255, 0, 0), 3);//������Բ����
		cv::circle(Eye, elPupil.center, 3, Scalar(0, 0, 255), -1, 8); //����ͫ������

		//cv::ellipse(ROI_iris, out.elPupil, pupiltracker::cvx::rgb(255, 0, 0), 3);//������Բ����
		//cv::circle(ROI_iris, out.pPupil, 3, Scalar(0, 0, 255), -1, 8); //����ͫ������

		pupil_results.push_back(pupil);
		pupil_results.back().at(0) = k;
		pupil_results.back().at(1) = out.pPupil.x;
		pupil_results.back().at(2) = out.pPupil.y;

		fprintf(pupil_center, "%d", k);
		fprintf(pupil_center, "\t%f", out.pPupil.x);
		fprintf(pupil_center, "\t%f\n", out.pPupil.y);

		//����ͫ�׼�����ı�����
		std::string postfix = to_string(k);
		if (postfix.size() < 4) postfix = "0" + postfix;//����Ϊ4λ��ʽ ��0824
		string pupildetection = pupil_prefix + whicheye + postfix + ".jpg"; //���ڵ�һ����Ƶpupil_prefixΪpupil_prefix_01,�ڶ���Ϊpupil_prefix_02
		
		//cv::imwrite(pupildetection, ROI_iris);
		cv::imwrite(pupildetection, Eye);
		//imshow("namedWindow", ROI_iris);
		//waitKey(1000);

	}
	return pupil_results;
}



//-----------------------------------��main( )������--------------------------------------------
//          ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼִ��
//-----------------------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{  
	double duration;
	clock_t start, end;
	start = clock();

	cv::Point eyeright, eyeleft, pos_marker;  //eyedetection ģ��ƥ���
	string video_name1, video_name2;
	video_name1 = Video_Name + str1+ video_format; //��"CT001_01.avi"
	video_name2 = Video_Name + str2 + video_format;
	string video1_dir = Video_Dir + str3 + Video_Name + str3 + video_name1;//����·��
	string video2_dir = Video_Dir + str3 + Video_Name + str3 + video_name2;
	string eyedetection = str4 + Video_Name + img_format; //�۾���������ͼ�񱣴�
	Mat eyetemplate_r = imread("D:\\yZheng2021\\6Projects\\01Strabismus\\PolyU\\Codes\\eyetemplate_r10.jpg", 1); //��1����������ģ��
	Mat eyetemplate_l = imread("D:\\yZheng2021\\6Projects\\01Strabismus\\PolyU\\Codes\\eyetemplate_l10.jpg", 1); //��1����������ģ��
	Mat marker = imread("D:\\yZheng2021\\6Projects\\01Strabismus\\PolyU\\Codes\\marker.jpg",0);//���뵲�����ģ��
	cv::VideoCapture cap1(video1_dir);  //������Ƶ1
	cv::VideoCapture cap2(video2_dir);  //������Ƶ2
	if (!cap1.isOpened()) {
		std::cout << "Unable to open the camera\n";
		std::exit(-1);
	}
	int fnum1 = cap1.get(7);   //��ȡ��Ƶ1��֡��
	int fnum2 = cap2.get(7);   //��ȡ��Ƶ1��֡��
	double position;
	
//**STEP1: eye detection ���������۾�ģ����ģ��ƥ��,���Լ������
	int num[10] = { 0 };  //����1~fnum1֮�䲻�ظ����������10��
	bool check;
	for (int i = 0; i<numframes; i++)
	{
		check = false;
		do
		{
			num[i] = randrange(1, fnum1);

			check = true;
			for (int j = 0; (check) && (j < i); j++) {
				check = (num[i] != num[j]);
			}
		} while (check == false);
	}	
	
	vector<cv::Point> eye_r(numframes), eye_l(numframes); //�洢matchpoint��λ��
	for (unsigned int i = 0; i < numframes; ++i)
	{
		position = num[i];
		cap1.set(CV_CAP_PROP_POS_FRAMES, position);
		cap1.read(frame); //��1������ͼ��
		Mat srcImage = frame; 

		//Rect left_region = Rect(0, 0, 0.5*srcImage.cols, srcImage.rows);
		//Rect right_region = Rect(0.5*srcImage.cols, 0, 0.5*srcImage.cols, srcImage.rows);  
		Rect left_region = Rect(0, 0.1*srcImage.rows, 0.5*srcImage.cols, 0.6*srcImage.rows);  //����������Сģ��ƥ�����õ��������Ұ�������0.2y~0.6y 0.1y~0.7y
		Rect right_region = Rect(0.5*srcImage.cols, 0.1*srcImage.rows, 0.5*srcImage.cols, 0.6*srcImage.rows);  
		Mat r_srcImage = srcImage(left_region);
		Mat l_srcImage = srcImage(right_region);

		eyeright = on_Matching(r_srcImage, eyetemplate_r, 5); //5��Ч����0��
		eyeleft = on_Matching(l_srcImage, eyetemplate_l, 5);  //5

		eyeright.y = eyeright.y + 0.1*srcImage.rows;
		eyeleft.x = 0.5*srcImage.cols + eyeleft.x;
		eyeleft.y = eyeleft.y + 0.1*srcImage.rows;

		eye_r.at(i) = eyeright;
		eye_l.at(i) = eyeleft;
		//��ԭͼ�ϻ��ƾ��ο�
		//rectangle(srcImage, eyeright, Point(eyeright.x + eyetemplate_r.cols, eyeright.y + eyetemplate_r.rows), Scalar(0, 0, 255), 2, 8, 0);
		////rectangle(srcimage, eyeleft, point(eyeleft.x + eyetemplate_l.cols, eyeleft.y + eyetemplate_l.rows), scalar(0, 0, 255), 2, 8, 0);
		//rectangle(srcImage, eyeleft, Point(eyeleft.x + eyetemplate_l.cols, eyeleft.y + eyetemplate_l.rows), Scalar(0, 0, 255), 2, 8, 0);
		////imshow("templatematching", srcimage);
		//std::string postfix = to_string( num[i] );
		//string eyedetection1 = str4 + Video_Name + "_" + postfix + img_format;
		//cv::imwrite(eyedetection1, srcImage);
	}
	
	sort(eye_r.begin(), eye_r.end(), compare_xy());  //��vector�е�matchpoint��������Ȼ���Ե�һ����Ϊ��㣬����룬������볬��ĳ����ֵ�������Ǹ������¼�������룬������������������ġ�
	sort(eye_l.begin(), eye_l.end(), compare_xy());
	double dist_r, dist_l;  //�����1��matchpoint��ǰһ���ľ���
	vector<unsigned int> segment_r, segment_l;  //���ڼ�¼matchpoint�����ķֶ�λ�ã�ÿһ���൱��һ����������
	for (unsigned int i = 1; i < numframes; ++i)
	{ 
		dist_r = sqrt((eye_r.at(i).x - eye_r.at(i-1).x)*(eye_r.at(i).x - eye_r.at(i-1).x) + (eye_r.at(i).y - eye_r.at(i-1).y)*(eye_r.at(i).y - eye_r.at(i-1).y));
		dist_l = sqrt((eye_l.at(i).x - eye_l.at(i-1).x)*(eye_l.at(i).x - eye_l.at(i-1).x) + (eye_l.at(i).y - eye_l.at(i-1).y)*(eye_l.at(i).y - eye_l.at(i-1).y));
		if (dist_r > 50)   segment_r.push_back(i);
		if (dist_l > 50)   segment_l.push_back(i);
	}

	cv::Point eyeright_vertex, eyeleft_vertex; //�۾�������ϽǶ�������
	cv::vector<unsigned int> votes_r(10), votes_l(10);
	double sum_xr = 0, sum_yr = 0, sum_xl = 0, sum_yl = 0;
	if (segment_r.size() == 0)  //˵��10֡��ƥ���Ƚ�һ�£�ֱ��ƽ��
	{
		for (unsigned int i = 0; i < numframes; ++i)
		{
			sum_xr += eye_r.at(i).x;
			sum_yr += eye_r.at(i).y;
		}
		eyeright_vertex.x = (double)1 / numframes *sum_xr;
		eyeright_vertex.y = (double)1 / numframes *sum_yr;
	}
	else  //�ֶ�>=����
	{
		votes_r[0] = segment_r[0];
		for (unsigned int j = 1; j < segment_r.size(); j++)   //segment���һ��Ԫ�ؾ��ǵ�һ�ε�ͶƱ����֮���һ��Ԫ��-ǰһ��Ԫ�صõ�����Ƭ�ε�ͶƱ�������һ��Ƭ��ͶƱ������numframes-Ԫ��ֵ��Ȼ��Ƚ��ĸ�ͶƱ�����
		{
			votes_r[j] = segment_r[j] - segment_r[j - 1];
		}
		votes_r[segment_r.size()] = numframes - segment_r.at(segment_r.size() - 1);
		auto maxPosition = max_element(votes_r.begin(), votes_r.end());  //�ҵ����ͶƱλ�ã����λ��������segment�ڼ���һһ��Ӧ
		int pr = std::distance(std::begin(votes_r), maxPosition);	//����pΪ1����votes[1]���λ��Ʊ����࣬����Ӧsegment_r��ĵ�2��
		if (pr == 0)
		{
			for (unsigned int i = 0; i < segment_r[pr]; ++i)
			{
				sum_xr += eye_r.at(i).x;
				sum_yr += eye_r.at(i).y;
			}
		}
		else if (pr == segment_r.size()) //��Ӧ���һ��Ƭ��ͶƱ���
		{
			for (unsigned int i = segment_r[pr - 1]; i < numframes; ++i)
			{
				sum_xr += eye_r.at(i).x;
				sum_yr += eye_r.at(i).y;
			}
		}
		else
		{
			for (unsigned int i = segment_r[pr - 1]; i < segment_r[pr]; ++i)
			{
				sum_xr += eye_r.at(i).x;
				sum_yr += eye_r.at(i).y;
			}
		}
		eyeright_vertex.x = (double)1 / (votes_r[pr])*sum_xr;
		eyeright_vertex.y = (double)1 / (votes_r[pr])*sum_yr;
	}

	if (segment_l.size() == 0)  //˵��10֡��ƥ���Ƚ�һ�£�ֱ��ƽ��
	{
		for (unsigned int i = 0; i < numframes; ++i)
		{
			sum_xl += eye_l.at(i).x;
			sum_yl += eye_l.at(i).y;
		}
		eyeleft_vertex.x = (double)1 / numframes *sum_xl;
		eyeleft_vertex.y = (double)1 / numframes *sum_yl;
	}
	else
	{
		votes_l[0] = segment_l[0];
		for (unsigned int j = 1; j < segment_l.size(); j++)   //segment���һ��Ԫ�ؾ��ǵ�һ�ε�ͶƱ����֮���һ��Ԫ��-ǰһ��Ԫ�صõ�����Ƭ�ε�ͶƱ�������һ��Ƭ��ͶƱ������numframes-Ԫ��ֵ��Ȼ��Ƚ��ĸ�ͶƱ�����
		{
			votes_l[j] = segment_l[j] - segment_l[j - 1];
		}
		votes_l[segment_l.size()] = numframes - segment_l.at(segment_l.size() - 1);
		auto maxPosition = max_element(votes_l.begin(), votes_l.end());  //�ҵ����ͶƱλ�ã����λ��������segment�ڼ���һһ��Ӧ
		int pl = std::distance(std::begin(votes_l), maxPosition);	//����pΪ1����votes[1]���λ��Ʊ����࣬����Ӧsegment_r��ĵ�2�Σ�����ط�Ҫ�ж�p1�Ƿ���0�������һ�Ρ�
		if(pl == 0)
		{
			for (unsigned int i = 0; i < segment_l[pl]; ++i)
			{
				sum_xl += eye_l.at(i).x;
				sum_yl += eye_l.at(i).y;
			}
		}
		else if (pl == segment_l.size()) //��Ӧ���һ��Ƭ��ͶƱ���
		{
			for (unsigned int i = segment_l[pl - 1]; i < numframes; ++i)
			{
				sum_xl += eye_l.at(i).x;
				sum_yl += eye_l.at(i).y;
			}
		}
		else
		{
			for (unsigned int i = segment_l[pl - 1]; i < segment_l[pl]; ++i)
			{
				sum_xl += eye_l.at(i).x;
				sum_yl += eye_l.at(i).y;
			}
		}
		eyeleft_vertex.x = (double)1 / (votes_l[pl])*sum_xl;
		eyeleft_vertex.y = (double)1 / (votes_l[pl])*sum_yl;
		
	}
	//eye region validation����  �������������overlap�������۵�x����-700�������۵Ľ��Ƽ�ࣩ��֮���غ���
	int area_overlap;
	int eyeleft_vertex_x_flip = eyeleft_vertex.x - 700; // 700 ��ȥ700���п���Ϊ��ֵ�����Լ�һ���ж�
	eyeleft_vertex_x_flip = (eyeleft_vertex_x_flip > 0) ? eyeleft_vertex_x_flip : 0;
	if (abs(eyeleft_vertex_x_flip - eyeright_vertex.x) >= eyetemplate_r.cols || abs(eyeleft_vertex.y - eyeright_vertex.y) >= eyetemplate_r.rows)
	{
		area_overlap = 0;//�����ۼ���û��overlap
	}
	else  //�ж�overlap�����
	{
		if (eyeleft_vertex_x_flip >= eyeright_vertex.x)
		{
			if (eyeleft_vertex.y >= eyeright_vertex.y)
				area_overlap = (eyeright_vertex.x + eyetemplate_r.cols - eyeleft_vertex_x_flip) * (eyeright_vertex.y + eyetemplate_r.rows - eyeleft_vertex.y);
			else
				area_overlap = (eyeright_vertex.x + eyetemplate_r.cols - eyeleft_vertex_x_flip) * (eyeleft_vertex.y + eyetemplate_r.rows - eyeright_vertex.y);
		}
		else
		{
			if (eyeleft_vertex.y >= eyeright_vertex.y)
				area_overlap = (eyeleft_vertex_x_flip + eyetemplate_r.cols - eyeright_vertex.x) * (eyeright_vertex.y + eyetemplate_r.rows - eyeleft_vertex.y);
			else
				area_overlap = (eyeleft_vertex_x_flip + eyetemplate_r.cols - eyeright_vertex.x) * (eyeleft_vertex.y + eyetemplate_r.rows - eyeright_vertex.y);
		}
	}
	//���overlapС��ĳ����ֵ��˵�������ۼ��򲻷����۾�����ˮƽ�ԳƵ����飬matchpoint������
	if ((double)area_overlap / (eyetemplate_r.cols * eyetemplate_r.rows) > 0.25)
	{ //����matchpoint������
	}
	else  //�Լ�����ȶ���Ϊ׼,���ۼ�����Ƚ��ȶ�
	{
		eyeright_vertex.x = eyeleft_vertex_x_flip;
		eyeright_vertex.y = eyeleft_vertex.y;
	}
	//��ʾ������eye detection���
	//cap1.set(CV_CAP_PROP_POS_FRAMES, 120);
	//cap1.read(frame);
	//rectangle(frame, eyeright_vertex, Point(eyeright_vertex.x + eyetemplate_r.cols, eyeright_vertex.y + eyetemplate_r.rows), Scalar(0, 0, 255), 2, 8, 0);
	//rectangle(frame, eyeleft_vertex, Point(eyeleft_vertex.x + eyetemplate_r.cols, eyeleft_vertex.y + eyetemplate_r.rows), Scalar(0, 0, 255), 2, 8, 0);
	//cv::imwrite(eyedetection, frame);

	pos_right = Rect(eyeright_vertex.x, eyeright_vertex.y, eyetemplate_r.cols, eyetemplate_r.rows);  //����λ��
	pos_left = Rect(eyeleft_vertex.x, eyeleft_vertex.y, eyetemplate_r.cols, eyetemplate_r.rows);  //����λ��

//**STEP2: Iris boundary detection
	
	IrisTracker::findIrisCircle_out out_irisr, out_irisl;
	IrisTracker::TrackerParams params_iris;
	params_iris.Conduct_HaarSurrounded = false;
	params_iris.Radius_Min = 56;//(56,80)for Strabismusvideos
	params_iris.Radius_Max = 80;//
	params_iris.scaling = 0.6;
	params_iris.lowThres = 0.03;  //0.10
	params_iris.highThres = 0.08;  //0.15

	Mat Eye_right_gray, Eye_left_gray;
	Masek masek;
	int radius_r[numframes] = { 0 };
	int radius_l[numframes] = { 0 }; //�����������飬����ʢ�Ű뾶ֵ
	vector<cv::Point> iris_r(numframes), iris_l(numframes); //�洢��Ĥ���ĵ�λ��
	for (int i = 0; i < numframes; ++i)
	{
		position = num[i];
		//position = i + 1;
		cap1.set(CV_CAP_PROP_POS_FRAMES, position);
		cap1.read(frame); //��1������ͼ��
		Eye_right = frame(pos_right);
		Eye_left = frame(pos_left);

		IrisTracker::findIrisCircle(params_iris, Eye_right, out_irisr);//�ڸ���Ȥ���۾������Ͻ���ͫ�׼��
		IrisTracker::findIrisCircle(params_iris, Eye_left, out_irisl);//�ڸ���Ȥ���۾������Ͻ���ͫ�׼��
		
		//��ʾ������iris localization���
		cv::circle(Eye_right, out_irisr.pIris, 3, Scalar(0, 255, 0), -1, 8, 0);// circle center
		cv::circle(Eye_right, out_irisr.pIris, out_irisr.r_iris, Scalar(0, 255, 0), 3, 8, 0);// circle outline
		std::string postfix = to_string( num[i] );
		string irisdetection_r = str5 + Video_Name + "_" + postfix + img_format;
		cv::imwrite(irisdetection_r, Eye_right);
		radius_r[i] = out_irisr.r_iris;
		iris_r.at(i) = out_irisr.pIris;
		//��ʾ������iris localization���
		cv::circle(Eye_left, out_irisl.pIris, 3, Scalar(0, 255, 0), -1, 8, 0);// circle center
		cv::circle(Eye_left, out_irisl.pIris, out_irisl.r_iris, Scalar(0, 255, 0), 3, 8, 0);// circle outline
		string irisdetection_l = str6 + Video_Name + "_" + postfix + img_format;
		cv::imwrite(irisdetection_l, Eye_left);
		radius_l[i] = out_irisl.r_iris;
		iris_l.at(i) = out_irisl.pIris;
	}
	//���������õ���Ĥ�İ뾶
	radius_right = mostFrequent(radius_r, numframes);
	radius_left = mostFrequent(radius_l, numframes);
	int diameter_right = radius_right * 2;
	int diameter_left = radius_left * 2;
	//std::cout << "diameter_right: " << diameter_right << endl;
	//std::cout << "diameter_left: " << diameter_left << endl;

	sort(iris_r.begin(), iris_r.end(), compare_xy());  //��vector�е�iris���Ľ�������Ȼ���Ե�һ����Ϊ��㣬����룬������볬��ĳ����ֵ�������Ǹ������¼�������룬������������������ġ�
	sort(iris_l.begin(), iris_l.end(), compare_xy());
	double dist_irisr, dist_irisl;  //�����1��iris center��ǰһ���ľ���
	vector<unsigned int> segment_irisr, segment_irisl;  //���ڼ�¼iris center�����ķֶ�λ�ã�ÿһ���൱��һ����������
	for (unsigned int i = 1; i < numframes; ++i)
	{
		dist_irisr = sqrt((iris_r.at(i).x - iris_r.at(i - 1).x)*(iris_r.at(i).x - iris_r.at(i - 1).x) + (iris_r.at(i).y - iris_r.at(i - 1).y)*(iris_r.at(i).y - iris_r.at(i - 1).y));
		dist_irisl = sqrt((iris_l.at(i).x - iris_l.at(i - 1).x)*(iris_l.at(i).x - iris_l.at(i - 1).x) + (iris_l.at(i).y - iris_l.at(i - 1).y)*(iris_l.at(i).y - iris_l.at(i - 1).y));
		if (dist_irisr > 0.5*radius_right)   segment_irisr.push_back(i);
		if (dist_irisl > 0.5*radius_left)   segment_irisl.push_back(i);
	}

	cv::vector<unsigned int> votes_irisr(10), votes_irisl(10);
	double sum_irisxr = 0, sum_irisyr = 0, sum_irisxl = 0, sum_irisyl = 0;
	if (segment_irisr.size() == 0)  //˵��10֡��ƥ���Ƚ�һ�£�ֱ��ƽ��
	{
		for (unsigned int i = 0; i < numframes; ++i)
		{
			sum_irisxr += iris_r.at(i).x;
			sum_irisyr += iris_r.at(i).y;
		}
		irisright_center.x = (double)1 / numframes *sum_irisxr;
		irisright_center.y = (double)1 / numframes *sum_irisyr;
	}
	else  //�ֶ�>=����
	{
		votes_irisr[0] = segment_irisr[0];
		for (unsigned int j = 1; j < segment_irisr.size(); j++)   //segment���һ��Ԫ�ؾ��ǵ�һ�ε�ͶƱ����֮���һ��Ԫ��-ǰһ��Ԫ�صõ�����Ƭ�ε�ͶƱ�������һ��Ƭ��ͶƱ������numframes-Ԫ��ֵ��Ȼ��Ƚ��ĸ�ͶƱ�����
		{
			votes_irisr[j] = segment_irisr[j] - segment_irisr[j - 1];
		}
		votes_irisr[segment_irisr.size()] = numframes - segment_irisr.at(segment_irisr.size() - 1);
		auto maxPosition = max_element(votes_irisr.begin(), votes_irisr.end());  //�ҵ����ͶƱλ�ã����λ��������segment�ڼ���һһ��Ӧ
		int pr = std::distance(std::begin(votes_irisr), maxPosition);	//����pΪ1����votes[1]���λ��Ʊ����࣬����Ӧsegment_r��ĵ�2��
		if (pr == 0)
		{
			for (unsigned int i = 0; i < segment_irisr[pr]; ++i)
			{
				sum_irisxr += iris_r.at(i).x;
				sum_irisyr += iris_r.at(i).y;
			}
		}
		else if (pr == segment_irisr.size()) //��Ӧ���һ��Ƭ��ͶƱ���
		{
			for (unsigned int i = segment_irisr[pr - 1]; i < numframes; ++i)
			{
				sum_irisxr += iris_r.at(i).x;
				sum_irisyr += iris_r.at(i).y;
			}
		}
		else
		{
			for (unsigned int i = segment_irisr[pr - 1]; i < segment_irisr[pr]; ++i)
			{
				sum_irisxr += iris_r.at(i).x;
				sum_irisyr += iris_r.at(i).y;
			}
		}
		irisright_center.x = (double)1 / (votes_irisr[pr]) *sum_irisxr;
		irisright_center.y = (double)1 / (votes_irisr[pr]) *sum_irisyr;
	}

	if (segment_irisl.size() == 0)  //˵��10֡��ƥ���Ƚ�һ�£�ֱ��ƽ��
	{
		for (unsigned int i = 0; i < numframes; ++i)
		{
			sum_irisxl += iris_l.at(i).x;
			sum_irisyl += iris_l.at(i).y;
		}
		irisleft_center.x = (double)1 / numframes *sum_irisxl;
		irisleft_center.y = (double)1 / numframes *sum_irisyl;
	}
	else
	{
		votes_irisl[0] = segment_irisl[0];
		for (unsigned int j = 1; j < segment_irisl.size(); j++)   //segment���һ��Ԫ�ؾ��ǵ�һ�ε�ͶƱ����֮���һ��Ԫ��-ǰһ��Ԫ�صõ�����Ƭ�ε�ͶƱ�������һ��Ƭ��ͶƱ������numframes-Ԫ��ֵ��Ȼ��Ƚ��ĸ�ͶƱ�����
		{
			votes_irisl[j] = segment_irisl[j] - segment_irisl[j - 1];
		}
		votes_irisl[segment_irisl.size()] = numframes - segment_irisl.at(segment_irisl.size() - 1);
		auto maxPosition = max_element(votes_irisl.begin(), votes_irisl.end());  //�ҵ����ͶƱλ�ã����λ��������segment�ڼ���һһ��Ӧ
		int pl = std::distance(std::begin(votes_irisl), maxPosition);	//����pΪ1����votes[1]���λ��Ʊ����࣬����Ӧsegment_r��ĵ�2�Σ�����ط�Ҫ�ж�p1�Ƿ���0�������һ�Ρ�
		if (pl == 0)
		{
			for (unsigned int i = 0; i < segment_irisl[pl]; ++i)
			{
				sum_irisxl += iris_l.at(i).x;
				sum_irisyl += iris_l.at(i).y;
			}
		}
		else if (pl == segment_irisl.size()) //��Ӧ���һ��Ƭ��ͶƱ���
		{
			for (unsigned int i = segment_irisl[pl - 1]; i < numframes; ++i)
			{
				sum_irisxl += iris_l.at(i).x;
				sum_irisyl += iris_l.at(i).y;
			}
		}
		else
		{
			for (unsigned int i = segment_irisl[pl - 1]; i < segment_irisl[pl]; ++i)
			{
				sum_irisxl += iris_l.at(i).x;
				sum_irisyl += iris_l.at(i).y;
			}
		}
		irisleft_center.x = (double)1 / (votes_irisl[pl])*sum_irisxl;
		irisleft_center.y = (double)1 / (votes_irisl[pl])*sum_irisyl;

	}
	


//**STEP3: Keyframe detection	
	vector< vector<int> > tm_results;// declare 2D vector
	vector<int> tm_frame(3, 0);// make new row (arbitrary example)
	string str7 = "tm_";
	string name_tmresults = str7 + Video_Name + ".txt";
	FILE* TM_results = fopen(name_tmresults.c_str(), "w");//��ͫ�����ļ�������浽pupilcenter.txt�ļ���
	Mat upper_srcImage;
	Rect upper_region;
	vector<int> keyframe(12);
	if (Conduct_keyframe_search == FALSE){
		keyframe = { 125, 455, 785, 1225, 1555, 1885, 2145, 2260, 2375, 2490, 2605, 2720 };		
	}
	else
	{	//ͨ��markerģ�����ȡ�����λ��	
		for (int i = 0; i < fnum1; ++i)
		{
			position = i;    //C++��Ƶλ�ô�0��ʼ
			cap1.set(CV_CAP_PROP_POS_FRAMES, position);
			
			try {
			cap1.read(frame); //��1������ͼ��
			cvtColor(frame, frame, CV_BGR2GRAY);//����ͨ��BGRתΪ��ͨ��gray
			Mat srcImage = frame;
			upper_region = Rect(0, 0, srcImage.cols, 0.33 * srcImage.rows);//0.25
			upper_srcImage = srcImage(upper_region); //marker��������face���ϰ벿������
			pos_marker = on_Matching(upper_srcImage, marker, 0); //5��Ч����0��

			tm_results.push_back(tm_frame);
			tm_results.back().at(0) = position;
			tm_results.back().at(1) = pos_marker.x;
			tm_results.back().at(2) = pos_marker.y;
			}
			catch(Exception e) {
				continue;
			}
		}
		for (int i = 0; i < fnum2; ++i)
		{
			position = i;
			cap2.set(CV_CAP_PROP_POS_FRAMES, position);
			cap2.read(frame); //��1������ͼ��
			cvtColor(frame, frame, CV_BGR2GRAY);//����ͨ��BGRתΪ��ͨ��gray
			Mat srcImage = frame;
			upper_region = Rect(0, 0, srcImage.cols, 0.33 * srcImage.rows);
			upper_srcImage = srcImage(upper_region); //marker��������face���ϰ벿������
			pos_marker = on_Matching(upper_srcImage, marker, 0); //5��Ч����0��

			tm_results.push_back(tm_frame);
			tm_results.back().at(0) = position + fnum1;
			tm_results.back().at(1) = pos_marker.x;
			tm_results.back().at(2) = pos_marker.y;
		}
		//tm_results����Ϊtxt�ļ�
		for (int i = 0; i < tm_results.size(); i++)
		{
				fprintf(TM_results, "%d", i);
				fprintf(TM_results, "\t%d", tm_results.at(i).at(1));
				fprintf(TM_results, "\t%d\n", tm_results.at(i).at(2));
		}
		fclose(TM_results);
		keyframe = keyframe_search(tm_results, 20); //ͨ��keyframe_search�㷨��ģ��ƥ�����ϵõ��ؼ�֡
	}
		
	vector<int> keyframe(12);
	vector<vector<int>> tms;
	vector<int> tm_row(3, 0);
	ifstream readtm;
	readtm.open("D:\\yZheng2021\\6Projects\\01Strabismus\\����\\Pupil detection\\�Ƚ��㷨\\5Our method\\pupildetection_forOURS\\Algorithm_Swirki\\tm_dataresults_c++\\tm_CT059.txt");
	string line;
	while (getline(readtm, line)) {
		istringstream iss(line);
		tms.push_back(tm_row);
		iss >> tms.back().at(0) >> tms.back().at(1) >> tms.back().at(2);

	}
	keyframe = keyframe_search(tms, 20);
	
//**STEP4: Pupil detection
	params.Radius_Min = 20;//(20,45)for strabismusvideos
	params.Radius_Max = 45;

	params.Conduct_Gamma = false;
	params.Conduct_HaarRectangle = false;
	params.CannyBlur = 1;
	params.CannyThreshold1 = 30;
	params.CannyThreshold2 = 50;
	params.StarburstPoints = 0;

	params.PercentageInliers = 30;
	params.InlierIterations = 2;
	params.ImageAwareSupport = true;
	params.EarlyTerminationPercentage = 95;
	params.EarlyRejection = true;
	params.Seed = -1;

	cv::Rect roi_right((irisright_center.x - radius_right >= 0) ? irisright_center.x - radius_right : 0, (irisright_center.y - radius_right >= 0) ? irisright_center.y - radius_right : 0, (irisright_center.x + radius_right >= pos_right.width) ? pos_right.width - irisright_center.x + radius_right : 2 * radius_right, (irisright_center.y + radius_right >= pos_right.height) ? pos_right.height - irisright_center.y + radius_right : 2 * radius_right);
	cv::Rect roi_left((irisleft_center.x - radius_left >= 0) ? irisleft_center.x - radius_left : 0, (irisleft_center.y - radius_left >= 0) ? irisleft_center.y - radius_left : 0, (irisleft_center.x + radius_left >= pos_left.width) ? pos_left.width - irisleft_center.x + radius_left : 2 * radius_left, (irisleft_center.y + radius_left >= pos_left.height) ? pos_left.height - irisleft_center.y + radius_left : 2 * radius_left);


	vector< vector<float> > CoverRight1, CoverRight2, CoverRight3, CoverLeft1, CoverLeft2, CoverLeft3, AlterRight1, AlterLeft1, AlterRight2, AlterLeft2, AlterRight3, AlterLeft3;
	// make new row (arbitrary example)
	vector<float> pupil(3, 0);
	
	
	CoverRight1 = pupil_detection(cap1, keyframe[0], pos_left, roi_left, pupil_prefix_01, "_l");//���������ۣ�������
	CoverRight2 = pupil_detection(cap1, keyframe[1], pos_left, roi_left, pupil_prefix_01, "_l");
	CoverRight3 = pupil_detection(cap1, keyframe[2], pos_left, roi_left, pupil_prefix_01, "_l");
	CoverLeft1 = pupil_detection(cap1, keyframe[3], pos_right, roi_right, pupil_prefix_01, "_r");
	CoverLeft2 = pupil_detection(cap1, keyframe[4], pos_right, roi_right, pupil_prefix_01, "_r");//��Ƶ01
	//CoverLeft3 = pupil_detection(cap1, keyframe[5], pos_right, roi_right, pupil_prefix_02, "_r");  //CT039,CT040,CT041,CT042
	CoverLeft3 = pupil_detection(cap2, keyframe[5] - fnum1, pos_right, roi_right, pupil_prefix_02, "_r");
	AlterRight1 = pupil_detection(cap2, keyframe[6] - fnum1, pos_left, roi_left, pupil_prefix_02, "_l");//������ס���ۣ�������
	AlterLeft1 = pupil_detection(cap2, keyframe[7] - fnum1, pos_right, roi_right, pupil_prefix_02, "_r");
	AlterRight2 = pupil_detection(cap2, keyframe[8] - fnum1, pos_left, roi_left, pupil_prefix_02, "_l");
	AlterLeft2 = pupil_detection(cap2, keyframe[9] - fnum1, pos_right, roi_right, pupil_prefix_02, "_r");
	AlterRight3 = pupil_detection(cap2, keyframe[10] - fnum1, pos_left, roi_left, pupil_prefix_02, "_l");
	AlterLeft3 = pupil_detection(cap2, keyframe[11] - fnum1, pos_right, roi_right, pupil_prefix_02, "_r");//��Ƶ02*/

	fclose(pupil_center);

//**STEP5: deviation calculationƫ��������
	vector< vector<float> > final_CR1, final_CR2, final_CR3, final_CL1, final_CL2, final_CL3, final_AR1, final_AL1, final_AR2, final_AL2, final_AR3, final_AL3;
	final_CR1 = delete_row_interation(CoverRight1, 1, 2);
	final_CR2 = delete_row_interation(CoverRight2, 1, 2);
	final_CR3 = delete_row_interation(CoverRight3, 1, 2);
	final_CL1 = delete_row_interation(CoverLeft1, 1, 2);
	final_CL2 = delete_row_interation(CoverLeft2, 1, 2);
	final_CL3 = delete_row_interation(CoverLeft3, 1, 2);

	final_AR1 = delete_row_interation(AlterRight1, 1, 2);
	final_AL1 = delete_row_interation(AlterLeft1, 1, 2);
	final_AR2 = delete_row_interation(AlterRight2, 1, 2);
	final_AL2 = delete_row_interation(AlterLeft2, 1, 2);
	final_AR3 = delete_row_interation(AlterRight3, 1, 2);
	final_AL3 = delete_row_interation(AlterLeft3, 1, 2);

	cv::Point2f dev_cr1, dev_cr2, dev_cr3, dev_cl1, dev_cl2, dev_cl3, dev_ar1, dev_al1, dev_ar2, dev_al2, dev_ar3, dev_al3;  //���μ���ƫ����delta_x,delta_y
	dev_cr1 = deviation(final_CR1, 1, 2);
	dev_cr2 = deviation(final_CR2, 1, 2);
	dev_cr3 = deviation(final_CR3, 1, 2);
	dev_cl1 = deviation(final_CL1, 1, 2);
	dev_cl2 = deviation(final_CL2, 1, 2);
	dev_cl3 = deviation(final_CL3, 1, 2);
	dev_ar1 = deviation(final_AR1, 1, 2);
	dev_al1 = deviation(final_AL1, 1, 2);
	dev_ar2 = deviation(final_AR2, 1, 2);
	dev_al2 = deviation(final_AL2, 1, 2);
	dev_ar3 = deviation(final_AR3, 1, 2);
	dev_al3 = deviation(final_AL3, 1, 2); //�����ڸ�ʱ������δ�ڸǵ�һ��������ע�Ӷ������ƶ������Խ��������ۣ������������˶� Colum1 =1�� colum2 = 2

	float x_cr, y_cr, x_cl, y_cl, x_ar, y_ar, x_al, y_al;  //ÿ�ּ�����ʽ�����ڸǵ�ƽ��ƫ����
	//x_cr =  abs(dev_cr3.x);
	x_cr = (abs(dev_cr1.x) + abs(dev_cr2.x) + abs(dev_cr3.x)) / 3;
	y_cr = (abs(dev_cr1.y) + abs(dev_cr2.y) + abs(dev_cr3.y)) / 3;
	x_cl = (abs(dev_cl1.x) + abs(dev_cl2.x) + abs(dev_cl3.x)) / 3;
	y_cl = (abs(dev_cl1.y) + abs(dev_cl2.y) + abs(dev_cl3.y)) / 3;
	x_ar = (abs(dev_ar1.x) + abs(dev_ar2.x) + abs(dev_ar3.x)) / 3;
	y_ar = (abs(dev_ar1.y) + abs(dev_ar2.y) + abs(dev_ar3.y)) / 3;
	x_al = (abs(dev_al1.x) + abs(dev_al2.x) + abs(dev_al3.x)) / 3;
	y_al = (abs(dev_al1.y) + abs(dev_al2.y) + abs(dev_al3.y)) / 3;

	float x_cr_diopter, y_cr_diopter, x_cl_diopter, y_cl_diopter, x_ar_diopter, y_ar_diopter, x_al_diopter, y_al_diopter;
	x_cr_diopter = strabismus_degree(x_cr, diameter_left);
	y_cr_diopter = strabismus_degree(y_cr, diameter_left);
	x_cl_diopter = strabismus_degree(x_cl, diameter_right);
	y_cl_diopter = strabismus_degree(y_cl, diameter_right);
	x_ar_diopter = strabismus_degree(x_ar, diameter_left);
	y_ar_diopter = strabismus_degree(y_ar, diameter_left);
	x_al_diopter = strabismus_degree(x_al, diameter_right);
	y_al_diopter = strabismus_degree(y_al, diameter_right);

//**step6: strabismus detection including strabismus or not, direction and angles
	float x_cover_diopter, y_cover_diopter, x_altcover_diopter, y_altcover_diopter;
	x_cover_diopter = 0.5*(x_cr_diopter + x_cl_diopter);
	y_cover_diopter = 0.5*(y_cr_diopter + y_cl_diopter);
	x_altcover_diopter = 0.5*(x_ar_diopter + x_al_diopter);

	std::cout << "horizontal deviation in diopter: " << x_altcover_diopter << endl;

	end = clock();
	duration = ( end - start ) / (double) CLOCKS_PER_SEC;  
	std::cout<<"time: "<< duration <<std::endl;        // ��ʾ����һ֡ͼ������ʱ��
	

 	std::cout << "The end" << std::endl;
	getchar();
	return 0;
}


