#include "stdafx.h"
#include <unordered_map>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "strabismus.h"

using namespace cv;
using namespace std;

//对iris半径检测值进行投票，出现频率最高的为iris的最终半径
int mostFrequent(int arr[], int n)
{
	// Insert all elements in hash.
	unordered_map<int, int> hash;
	for (int i = 0; i < n; i++)
		hash[arr[i]]++;

	// find the max frequency
	int max_count = 0, res = -1;
	for (auto i : hash) {
		if (max_count < i.second) {
			res = i.first;
			max_count = i.second;
		}
	}

	return res;
}

//函数subtract用于计算输入矩阵A前一行（帧号，X，Y）与后一行之差，X超过一定阈值所对应的行删掉，最后返回删除行之后的矩阵和删除的行数deleted_row
void subtract(vector<vector<float>> A, int colum1, int colum2, subtract_out& s_out)
{
	int v = 5;//v为两帧之间瞳孔坐标x方向的差值的阈值
	vector<vector<float>> B;
	vector<float> row(4, 0);
	for (int j = 0; j < A.size() - 1; j++)
	{
		B.push_back(row);
		B.back().at(0) = A.at(j).at(0); //B的行中第一个元素存储当前帧的帧号码
		B.back().at(1) = abs(A.at(j + 1).at(0) - A.at(j).at(0)); //后一帧与前一阵的帧号之差delta_f
		B.back().at(2) = abs(A.at(j + 1).at(colum1) - A.at(j).at(colum1)); //colum1为1，求的是右眼瞳孔位置的delta_x 
		B.back().at(3) = abs(A.at(j + 1).at(colum2) - A.at(j).at(colum2)); //colum2为2，求的是右眼瞳孔位置的delta_y
	}
	vector<int> C, D; //找出B中X差值超过10pixel对应的帧，将帧号保存在C中，D存储的是行号
	for (int k = 0; k < B.size(); k++)
	{
		if ((B.at(k).at(2) >= v * B.at(k).at(1)) || (B.at(k).at(3) >= v * B.at(k).at(1)))
		{
			C.push_back(B.at(k).at(0) + 1);
			D.push_back(k + 1);//D中存的数对应要删除的行号
		}
	}
	vector<vector<float>> copy_A = A;   //删除原数据副本中对应帧号的行
	for (int i = D.size() - 1; i >= 0; i--)
	{
		copy_A.erase(copy_A.begin() + D[i]);
	}
	int deleted_row = C.size();
	s_out.copy_A = copy_A;
	s_out.deleted_row = deleted_row;
}

//delete_row_interation用于在计算偏移量时迭代的删除单次激励瞳孔坐标矩阵A中瞳孔检测奇异值
vector<vector<float>> delete_row_interation(vector<vector<float>> A, int colum1, int colum2)
{
	vector< vector<float> > final_A;
	subtract_out s_out;
	subtract(A, colum1, colum2, s_out);
	int loop_count = 1;
	while (s_out.deleted_row != 0)
	{
		subtract(s_out.copy_A, colum1, colum2, s_out);
		loop_count++;
		if (loop_count > 10)
			break;
	}
	final_A = s_out.copy_A;
	return final_A;
}

//函数deviation()计算偏移量
cv::Point2f deviation(vector<vector<float>> A, int colum1, int colum2)
{
	float addx1 = 0; float addy1 = 0; float addx2 = 0; float addy2 = 0;
	for (int i = 1; i < 6; i++)
	{
		addx1 += A.at(i).at(colum1);
		addy1 += A.at(i).at(colum2);
	}
	for (int j = A.size() - 5; j < A.size(); j++)
	{
		addx2 += A.at(j).at(colum1);
		addy2 += A.at(j).at(colum2);
	}
cv:Point2f dev;
	dev.x = (addx2 - addx1) / 5;
	dev.y = (addy2 - addy1) / 5;
	return dev;
}

//函数strabismus_degree将像素偏移量转化为三棱镜度
float strabismus_degree(float Dev_pixel, float DEp)
{
	int DEmm = 11;
	int dpMM = 15; //HR ratio具体多少可以讨论
	float Dev_diopter = (DEmm / DEp)*dpMM*Dev_pixel;

	return Dev_diopter;
}