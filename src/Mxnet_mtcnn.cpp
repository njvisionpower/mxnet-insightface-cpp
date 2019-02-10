#include "mxnet/c_predict_api.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

#include "mtcnn.hpp"
#include "mxnet_mtcnn.hpp"
#include "utils.hpp"
#include "mxnet-cpp/MxNetCpp.h"

static int LoadFile(const std::string & fname, std::vector<char>& buf)
{
	std::ifstream fs(fname, std::ios::binary | std::ios::in);

	if (!fs.good())
	{
		std::cerr << fname << " does not exist" << std::endl;
		return -1;
	}

	fs.seekg(0, std::ios::end);
	int fsize = fs.tellg();

	fs.seekg(0, std::ios::beg);
	buf.resize(fsize);
	fs.read(buf.data(), fsize);

	fs.close();

	return 0;

}



int MxNetMtcnn::LoadModule(const std::string &proto_model_dir)
{
	model_dir_ = proto_model_dir;

	/* Load the network. */
	RNet_ = LoadRNet(1);

	if (RNet_ == nullptr)
		return -1;

	ONet_ = LoadONet(1);

	if (ONet_ == nullptr)
		return -1;

	return 0;
}


MxNetMtcnn::~MxNetMtcnn(void)
{
	MXPredFree(RNet_);
	MXPredFree(ONet_);

	if(!PredictVec.empty())
	{
		for(int i=0; i<PredictVec.size(); i++)
		{
		    PredictorHandle PNet = PredictVec[i];
			MXPredFree(PNet);
		}
	}

}

void MxNetMtcnn::LoadPNet(int h, int w)
{
	std::string param_file = model_dir_ + "/det1-0001.params";
	std::string json_file = model_dir_ + "/det1-symbol.json";
}

PredictorHandle MxNetMtcnn::LoadMxNetModule(const std::string& param_file, const std::string& json_file,
	int batch, int channel, int input_h, int input_w)
{
	int64 start = cv::getTickCount();
	std::vector<char> param_buffer;
	std::vector<char> json_buffer;
	PredictorHandle pred_hnd;

	if (LoadFile(param_file, param_buffer)<0)
		return nullptr;

	if (LoadFile(json_file, json_buffer)<0)
		return nullptr;

	int device_type = 1;
	int dev_id = 0;
	mx_uint  num_input_nodes = 1;
	const char * input_keys[1];
	const mx_uint input_shape_indptr[] = { 0, 4 };
	const mx_uint input_shape_data[] = {
		static_cast<mx_uint>(batch),
		static_cast<mx_uint>(channel),
		static_cast<mx_uint>(input_h),
		static_cast<mx_uint>(input_w)
	};

	input_keys[0] = "data";

	MXPredCreate(json_buffer.data(),
		param_buffer.data(),
		param_buffer.size(),
		device_type,
		dev_id,
		num_input_nodes,
		input_keys,
		input_shape_indptr,
		input_shape_data,
		&pred_hnd
	);
	
	return pred_hnd;
}

void MxNetMtcnn::clearPredictVec()
{
    if(!PredictVec.empty())
    {
        for(int i=0; i<PredictVec.size(); i++)
		    MXPredFree(PredictVec[i]);
    }
}
void MxNetMtcnn::Detect(cv::Mat& orig_img, std::vector<face_box>& face_list)
{
	int64 tt1 = cv::getTickCount();
	cv::Mat img;

	orig_img.convertTo(img, CV_32FC3);

	img = (img - 127.5)*0.0078125;

	int img_h = img.rows;
	int img_w = img.cols;

	std::vector<scale_window> win_list;

	std::vector<face_box> total_pnet_boxes;
	std::vector<face_box> total_rnet_boxes;
	std::vector<face_box> total_onet_boxes;

	cal_pyramid_list(img_h, img_w, min_size_, factor_, win_list);

	if(img_h != pnet_h || img_w != pnet_w)
	{   
	    std::cout<<"reload pNet as img shape change..."<<std::endl;
	    clearPredictVec();
	    PredictVec.clear();
	    for(int i=0; i<win_list.size(); i++)
        {
            std::string param_file = model_dir_ + "/det1-0001.params";
	        std::string json_file = model_dir_ + "/det1-symbol.json";
         
	        PredictorHandle pred = LoadMxNetModule(param_file, json_file, 1, 3, win_list[i].h, win_list[i].w);
            PredictVec.push_back(pred);
	    }
		pnet_h = img_h;
		pnet_w = img_w;
	}

	// TODO: can use multiple thread for every scale predictor
	for (int i = 0;i<win_list.size();i++)
	{
		std::vector<face_box>boxes;
        PredictorHandle& pred_pNet = PredictVec[i];
		RunPNet(img, win_list[i], boxes, pred_pNet);
		total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
	}

	std::vector<face_box> pnet_boxes;
	process_boxes(total_pnet_boxes, img_h, img_w, pnet_boxes);

	if (pnet_boxes.size() == 0)
		return;

	if (pnet_boxes.size()>rnet_batch_bound_)
	{
		RunRNet(img, pnet_boxes, total_rnet_boxes);
	}
	else
	{
		for (unsigned int i = 0;i<pnet_boxes.size();i++)
		{
			face_box out_box;
			if (RunPreLoadRNet(img, pnet_boxes[i], out_box)<0)
				continue;
			total_rnet_boxes.push_back(out_box);
		}
	}

	std::vector<face_box> rnet_boxes;
	process_boxes(total_rnet_boxes, img_h, img_w, rnet_boxes);

	if (rnet_boxes.size() == 0)
		return;

	if (rnet_boxes.size()>onet_batch_bound_)
	{
		RunONet(img, rnet_boxes, total_onet_boxes);
	}
	else
	{
		for (unsigned int i = 0;i<rnet_boxes.size();i++)
		{
			face_box out_box;
			if (RunPreLoadONet(img, rnet_boxes[i], out_box)<0)
				continue;
			total_onet_boxes.push_back(out_box);
		}
	}

	//calculate the landmark
	cal_landmark(total_onet_boxes);

	//Get Final Result
	regress_boxes(total_onet_boxes);
	nms_boxes(total_onet_boxes, 0.7, NMS_MIN, face_list);

}

void MxNetMtcnn::RunPNet(const cv::Mat& img, scale_window& win, std::vector<face_box>&box_list, PredictorHandle& PNet_)
{
	
	int scale_h = win.h;
	int scale_w = win.w;
	float scale = win.scale;
	cv::Mat  resized;
	cv::resize(img, resized, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_LINEAR);

	std::vector<float> input(3 * scale_h*scale_w);
	std::vector<cv::Mat> input_channels;
	set_input_buffer(input_channels, input.data(), scale_h, scale_w);
	cv::split(resized, input_channels);

	MXPredSetInput(PNet_, "data", input.data(), input.size());
	MXPredForward(PNet_);
	
	mx_uint *shape = NULL;
	mx_uint shape_len = 0;

	MXPredGetOutputShape(PNet_, 0, &shape, &shape_len);

	int reg_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		reg_size *= shape[i];

	MXPredGetOutputShape(PNet_, 1, &shape, &shape_len);

	int confidence_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		confidence_size *= shape[i];

	std::vector<float> reg(reg_size);
	std::vector<float> confidence(confidence_size);

	MXPredGetOutput(PNet_, 0, reg.data(), reg_size);
	MXPredGetOutput(PNet_, 1, confidence.data(), confidence_size);

	std::vector<face_box>  candidate_boxes;

	int feature_h = shape[2];
	int feature_w = shape[3];

	generate_bounding_box(confidence.data(), confidence.size(), reg.data(), scale, pnet_threshold_, feature_h, feature_w, candidate_boxes, false);

	nms_boxes(candidate_boxes, 0.5, NMS_UNION, box_list);

}

void MxNetMtcnn::CopyOnePatch(const cv::Mat& img, face_box&input_box, float * data_to, int height, int width)
{
	std::vector<cv::Mat> channels;
	set_input_buffer(channels, data_to, height, width);

	int pad_top = std::abs(input_box.py0 - input_box.y0);
	int pad_left = std::abs(input_box.px0 - input_box.x0);
	int pad_bottom = std::abs(input_box.py1 - input_box.y1);
	int pad_right = std::abs(input_box.px1 - input_box.x1);

	cv::Mat chop_img = img(cv::Range(input_box.py0, input_box.py1),
	cv::Range(input_box.px0, input_box.px1));

	cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::resize(chop_img, chop_img, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

	cv::split(chop_img, channels);
}

PredictorHandle MxNetMtcnn::LoadRNet(int batch)
{
	std::string param_file = model_dir_ + "/det2-0001.params";
	std::string json_file = model_dir_ + "/det2-symbol.json";

	return LoadMxNetModule(param_file, json_file, batch, 3, 24, 24);
}


PredictorHandle MxNetMtcnn::LoadONet(int batch)
{
	std::string param_file = model_dir_ + "/det3-0001.params";
	std::string json_file = model_dir_ + "/det3-symbol.json";

	return LoadMxNetModule(param_file, json_file, batch, 3, 48, 48);
}

void MxNetMtcnn::RunRNet(const cv::Mat& img, std::vector<face_box>& pnet_boxes,std::vector<face_box>& output_boxes)
{
	int batch=pnet_boxes.size();
	int input_channel = 3;
	int input_width = 24;
	int input_height = 24;
	int input_size=batch*input_channel*input_width*input_height;

	PredictorHandle rnet = LoadRNet(batch);

	if(rnet == nullptr)
		return ;

	/* load the data */
	std::vector<float> input(input_size);
	float * input_data=input.data();

	for(int i=0;i<batch;i++)
	{
		int patch_size=input_width*input_height*input_channel;

		CopyOnePatch(img,pnet_boxes[i], input_data,input_height,input_width);

		input_data+=patch_size;
	}

	MXPredSetInput(rnet,"data",input.data(),input_size);		
	MXPredForward(rnet);

	mx_uint *shape = NULL;
	mx_uint shape_len = 0;

	MXPredGetOutputShape(rnet,0,&shape,&shape_len);
	int reg_size=1;

	for(unsigned int i=0;i<shape_len;i++)
		reg_size*=shape[i];

	MXPredGetOutputShape(rnet,1,&shape,&shape_len);
	int confidence_size=1;

	for(unsigned int i=0;i<shape_len;i++)
		confidence_size*=shape[i];

	std::vector<float> reg(reg_size);
	std::vector<float> confidence(confidence_size);

	MXPredGetOutput(rnet,0,reg.data(),reg_size);
	MXPredGetOutput(rnet,1,confidence.data(),confidence_size);


	const float* confidence_data = confidence.data();
	const float* reg_data = reg.data();


	/* filter output now */
	int conf_page_size=confidence_size/batch;
	int reg_page_size=reg_size/batch;

	for(int i=0;i<batch;i++)
	{

		if (*(confidence_data+1) > rnet_threshold_){

			face_box output_box;
			face_box& input_box=pnet_boxes[i];

			output_box.x0=input_box.x0;
			output_box.y0=input_box.y0;
			output_box.x1=input_box.x1;
			output_box.y1=input_box.y1;

			output_box.score = *(confidence_data+1);

			output_box.regress[0]=reg_data[0];
			output_box.regress[1]=reg_data[1];
			output_box.regress[2]=reg_data[2];
			output_box.regress[3]=reg_data[3];

			output_boxes.push_back(output_box);

		}

		confidence_data+=conf_page_size;
		reg_data+=reg_page_size;
	}

	MXPredFree(rnet);
}


int MxNetMtcnn::RunPreLoadRNet(const cv::Mat& img, face_box& input_box, face_box& output_box)
{
	int input_channels = 3;
	int input_width = 24;
	int input_height = 24;

	std::vector<float> input(input_channels*input_width*input_height);

	CopyOnePatch(img, input_box, input.data(), input_height, input_width);

	MXPredSetInput(RNet_, "data", input.data(), input.size());
	MXPredForward(RNet_);

	mx_uint *shape = NULL;
	mx_uint shape_len = 0;

	MXPredGetOutputShape(RNet_, 0, &shape, &shape_len);
	int reg_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		reg_size *= shape[i];

	MXPredGetOutputShape(RNet_, 1, &shape, &shape_len);
	int confidence_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		confidence_size *= shape[i];

	std::vector<float> reg(reg_size);
	std::vector<float> confidence(confidence_size);

	MXPredGetOutput(RNet_, 0, reg.data(), reg_size);
	MXPredGetOutput(RNet_, 1, confidence.data(), confidence_size);


	const float* confidence_data = confidence.data() + confidence.size() / 2;
	const float* reg_data = reg.data();

	if (*(confidence_data) > rnet_threshold_) {
		output_box.x0 = input_box.x0;
		output_box.y0 = input_box.y0;
		output_box.x1 = input_box.x1;
		output_box.y1 = input_box.y1;

		output_box.score = *(confidence_data);

		output_box.regress[0] = reg_data[0];
		output_box.regress[1] = reg_data[1];
		output_box.regress[2] = reg_data[2];
		output_box.regress[3] = reg_data[3];

		return 0;

	}

	return -1;

}

int MxNetMtcnn::RunPreLoadONet(const cv::Mat& img, face_box& input_box, face_box& output_box)
{
	int input_channels = 3;
	int input_width = 48;
	int input_height = 48;

	std::vector<float> input(input_channels*input_width*input_height);

	CopyOnePatch(img, input_box, input.data(), input_height, input_width);

	MXPredSetInput(ONet_, "data", input.data(), input.size());
	MXPredForward(ONet_);

	mx_uint *shape = NULL;
	mx_uint shape_len = 0;

	MXPredGetOutputShape(ONet_, 1, &shape, &shape_len);
	int reg_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		reg_size *= shape[i];

	MXPredGetOutputShape(ONet_, 0, &shape, &shape_len);
	int points_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		points_size *= shape[i];

	MXPredGetOutputShape(ONet_, 2, &shape, &shape_len);
	int confidence_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		confidence_size *= shape[i];

	std::vector<float> reg(reg_size);
	std::vector<float> points(points_size);
	std::vector<float> confidence(confidence_size);

	MXPredGetOutput(ONet_, 0, points.data(), points_size);
	MXPredGetOutput(ONet_, 1, reg.data(), reg_size);
	MXPredGetOutput(ONet_, 2, confidence.data(), confidence_size);


	const float* confidence_data = confidence.data() + confidence.size() / 2;
	const float* reg_data = reg.data();
	const float* points_data = points.data();

	if (*(confidence_data) > onet_threshold_) {

		output_box.x0 = input_box.x0;
		output_box.y0 = input_box.y0;
		output_box.x1 = input_box.x1;
		output_box.y1 = input_box.y1;

		output_box.score = *(confidence_data);

		output_box.regress[0] = reg_data[0];
		output_box.regress[1] = reg_data[1];
		output_box.regress[2] = reg_data[2];
		output_box.regress[3] = reg_data[3];


		for (int j = 0; j<5; j++) {
			output_box.landmark.x[j] = *(points_data + j);
			output_box.landmark.y[j] = *(points_data + j + 5);
		}


		return 0;

	}

	return -1;

}

void MxNetMtcnn::RunONet(const cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes)
{
	int batch = rnet_boxes.size();
	int input_channel = 3;
	int input_width = 48;
	int input_height = 48;
	int input_size = batch*input_channel*input_width*input_height;

	PredictorHandle onet = LoadONet(batch);

	if (onet == nullptr)
		return;

	/* load the data */
	std::vector<float> input(input_size);
	float * input_data = input.data();

	for (int i = 0;i<batch;i++)
	{
		int patch_size = input_width*input_height*input_channel;

		CopyOnePatch(img, rnet_boxes[i], input_data, input_height, input_width);

		input_data += patch_size;
	}


	MXPredSetInput(onet, "data", input.data(), input.size());
	MXPredForward(onet);

	mx_uint *shape = NULL;
	mx_uint shape_len = 0;

	MXPredGetOutputShape(onet, 1, &shape, &shape_len);
	int reg_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		reg_size *= shape[i];

	MXPredGetOutputShape(onet, 0, &shape, &shape_len);
	int points_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		points_size *= shape[i];

	MXPredGetOutputShape(onet, 2, &shape, &shape_len);
	int confidence_size = 1;

	for (unsigned int i = 0;i<shape_len;i++)
		confidence_size *= shape[i];

	std::vector<float> reg(reg_size);
	std::vector<float> points(points_size);
	std::vector<float> confidence(confidence_size);

	MXPredGetOutput(onet, 0, points.data(), points_size);
	MXPredGetOutput(onet, 1, reg.data(), reg_size);
	MXPredGetOutput(onet, 2, confidence.data(), confidence_size);

	const float* confidence_data = confidence.data();
	const float* reg_data = reg.data();
	const float* points_data = points.data();

	int reg_page_size = reg_size / batch;
	int confidence_page_size = confidence_size / batch;
	int points_page_size = points_size / batch;

	for (int i = 0;i<batch;i++)
	{

		if (*(confidence_data + 1) > onet_threshold_) {

			face_box output_box;
			face_box & input_box = rnet_boxes[i];

			output_box.x0 = input_box.x0;
			output_box.y0 = input_box.y0;
			output_box.x1 = input_box.x1;
			output_box.y1 = input_box.y1;

			output_box.score = *(confidence_data + 1);

			output_box.regress[0] = reg_data[0];
			output_box.regress[1] = reg_data[1];
			output_box.regress[2] = reg_data[2];
			output_box.regress[3] = reg_data[3];


			for (int j = 0; j<5; j++) {
				output_box.landmark.x[j] = *(points_data + j);
				output_box.landmark.y[j] = *(points_data + j + 5);
			}

			output_boxes.push_back(output_box);
		}

		reg_data += reg_page_size;
		confidence_data += confidence_page_size;
		points_data += points_page_size;

	}
}


