#include "stdafx.h"

#include "opencv2/objdetect/objdetect.hpp"

//#include "opencv2/videoio.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include <stdio.h>

using namespace std;

using namespace cv;

void detectAndDisplay(Mat frame);

//String face_cascade_name, eyes_cascade_name;
String  eyes_cascade_name = "D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";

//CascadeClassifier face_cascade;

CascadeClassifier eyes_cascade;

String window_name = "Capture - Face detection";

int main(int argc, const char** argv)

{

	//CommandLineParser parser(argc, argv,

		//"{eyes_cascade|D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml|}");//"{face_cascade|OpenCV/Win32/data/haarcascades/haarcascade_frontalface_alt.xml|}"

	

	//face_cascade_name = parser.get<string>("face_cascade");

	//eyes_cascade_name = parser.get<string>("eyes_cascade");

	VideoCapture capture;

	Mat frame;

	//Load the cascades

	//if (!face_cascade.load(face_cascade_name)){ printf("Error loading face cascade\n"); return -1; };

	if (!eyes_cascade.load(eyes_cascade_name)){ printf("Error loading eyes cascade\n"); return -1; };

	//Read the video stream

	capture.open("D:\\֣��2021\\6.�о�����\\01Strabismus\\PolyU\\PolyU_videos\\CT001\\CT001_01.avi");

	if (!capture.isOpened()) { printf("Error opening video capture\n"); return -1; }
	int num = 1;
	while (num < 100)

	{
		capture.read(frame);
		if (frame.empty())

		{

			printf("No captured frame. Break!");

			break;

		}

		// Apply the classifier to the frame

		detectAndDisplay(frame);

		char c = (char)waitKey(10);

		if (c == 27) { break; } // escape

		num++;

	}

	return 0;

}

void detectAndDisplay(Mat frame)

{

	//std::vector<Rect> faces;

	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces

	//face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	//for (size_t i = 0; i < faces.size(); i++)

	{

		//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		//Mat faceROI = frame_gray(faces[i]);

		std::vector<Rect> eyes;

		//In each face, detect eyes

		eyes_cascade.detectMultiScale(frame_gray, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)

		{

			//Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			Point eye_center(eyes[j].x + eyes[j].width / 2, eyes[j].y + eyes[j].height / 2);

			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);

			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);

		}

	}

	//Show what you got

	imshow(window_name, frame);

}