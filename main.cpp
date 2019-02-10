#include<stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mxnet/c_predict_api.h>
#include <math.h>

#include "face_align.hpp"
#include "mxnet_mtcnn.hpp"
#include "feature_extract.hpp"
#include "make_label.hpp"
#include "comm_lib.hpp"

void test_mtcnn()
{
	std::string mtcnn_model = "mtcnn_model";
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule(mtcnn_model);

	std::string feature_model = "feature_model";
	Mxnet_extract extract;
	extract.LoadExtractModule("feature_model/model-0000.params", "feature_model/model-symbol.json",1,3,112,112);

	cv::Mat img = cv::imread("image/stephanie-sun.jpg");
	std::vector<face_box> face_info;
	mtcnn.Detect(img, face_info);
	for (int i = 0; i<face_info.size();i++)
	{
		auto face = face_info[i];
		std::cout << "face location: x0=" << face.x0 << " y0=" << face.y0 << "x1=" << face.x1 << "y1=" << face.y1 << std::endl;
		std::cout << "face landmark: x0=" << face.landmark.x[0] << " x1=" << face.landmark.x[1] << "x2=" << face.landmark.x[2] << "x3=" << face.landmark.x[3] <<"x4="<<face.landmark.x[4]<< std::endl;
		std::cout << "face landmark: y0=" << face.landmark.y[0] << " y1=" << face.landmark.y[1] << "y2=" << face.landmark.y[2] << "y3=" << face.landmark.y[3] << "y4=" << face.landmark.y[4] << std::endl;

		for (int j = 0; j < 5; j++)
		{
			cv::Point p(face.landmark.x[j], face.landmark.y[j]);
			cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
		}

		cv::Point pt1(face.x0, face.y0);
		cv::Point pt2(face.x1, face.y1);
		cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0),2);
		cv::imshow("img", img);
		cv::waitKey(0);
	}

}

void test_make_label(std::string path)
{

	std::vector<std::string> imagePath;
	std::vector<std::string> imageLabel;

	getFiles(path, imagePath, imageLabel);
	for (auto file : imagePath)
	{
		std::cout << "file: " << file << std::endl;
	}

	//this function will generate features from images
	make_label(imagePath, imageLabel, "mtcnn_model", "feature_model/model-0000.params", "feature_model/model-symbol.json");
}

void test_camera()
{
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule("mtcnn_model");
	Mxnet_extract extract;
	extract.LoadExtractModule("feature_model/model-0000.params", "feature_model/model-symbol.json", 1, 3, 112, 112);

	//loading features
	cv::FileStorage fs("features.xml", cv::FileStorage::READ);
	cv::Mat features;
	fs["features"] >> features;

	//loading labels
	std::ifstream file("labels.txt");
	std::string t;
	while (std::getline(file, t)) {}

	std::vector<std::string> labels;
	SplitString(t, labels, " ");

	cv::VideoCapture cap;
	cap.open(0); 
	if (!cap.isOpened())
		return;

	cv::Mat frame;
	while (1)
	{
		cap >> frame;
		double start = static_cast<double>(cv::getTickCount());

		//recognition: detect + align + feature extract + classification
		recognition(mtcnn, extract, frame, features, labels);
		double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
		std::cout << "spent: " << time << "s " << std::endl;

		if (frame.empty())
			break;
		cv::imshow("frame", frame);
		cv::waitKey(1);
		
	}
	cap.release();
}


int main(int argc, char* argv[]) {

	//test_mtcnn();
	//std::string path = "D:\\cpp\\mxnet_cpp\\mxnet_cpp\\image";
	//test_make_label(path);

	test_camera();
	system("pause");
	return 0;
}