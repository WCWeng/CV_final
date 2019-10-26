#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

Mat getHistImg(const vector<Mat>&);
vector<Mat> histComputer(vector<Mat>&);
vector<Mat> colorTransfer(vector<Mat>&, vector<Mat>&);

int main(){
    namedWindow("Input", CV_WINDOW_AUTOSIZE);
	namedWindow("Input Histogram", CV_WINDOW_AUTOSIZE);
    namedWindow("Model", CV_WINDOW_AUTOSIZE);
	namedWindow("Model Histogram", CV_WINDOW_AUTOSIZE);
	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	namedWindow("Result Histogram", CV_WINDOW_AUTOSIZE);
	
    Mat input;
	Mat model;
	Mat result;
	vector<Mat> inputplane;
	vector<Mat> modelplane;
	vector<Mat> resultplane;
	vector<Mat> inputhist;
	vector<Mat> modelhist;
	vector<Mat> resulthist;

	input = imread("input.png");
	model = imread("model.png");
	if (!input.data || !model.data)
		return -1;

		cvtColor(input, input, CV_BGR2Lab);
		cvtColor(model, model, CV_BGR2Lab);

		//Split frame to l, a, b plane.
		split(input, inputplane);
		split(model, modelplane);
		//Compute histogram from lab plane.
		inputhist = histComputer(inputplane);
		modelhist = histComputer(modelplane);

		resultplane = colorTransfer(inputplane, modelplane);
		resulthist = histComputer(resultplane);
		merge(resultplane, result);

		cvtColor(input, input, CV_Lab2BGR);
		cvtColor(model, model, CV_Lab2BGR);
		cvtColor(result, result, CV_Lab2BGR);
		
		imshow("Input", input);
		imshow("Model", model);
		imshow("Result", result);
		imshow("Input Histogram", getHistImg(inputhist));
		imshow("Model Histogram", getHistImg(modelhist));
		imshow("Result Histogram", getHistImg(resulthist));
		imwrite("result.png", result);
		
        waitKey(0);
    

    return 0;
}

Mat getHistImg(const vector<Mat>& lab_hist) {
	int histSize = 256;
	int hist_w = 400, hist_h = 300;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImg(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat l_hist, a_hist, b_hist;

	normalize(lab_hist[0], l_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	normalize(lab_hist[1], a_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	normalize(lab_hist[2], b_hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(l_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(l_hist.at<float>(i))),
			Scalar(255, 0, 0));
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(a_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(a_hist.at<float>(i))),
			Scalar(0, 255, 0));
		line(histImg, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(0, 0, 255));
	}

	return histImg;
}

vector<Mat> histComputer(vector<Mat>& lab_planes) {
	vector<Mat> lab_hist;
	int histSize = 256;
	float range[] = {0, 255};
	const float* histRange = {range};
	Mat l_hist, a_hist, b_hist;

	calcHist(&lab_planes[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange);
	calcHist(&lab_planes[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange);
	calcHist(&lab_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);

	lab_hist.push_back(l_hist);
	lab_hist.push_back(a_hist);
	lab_hist.push_back(b_hist);

	return lab_hist;
}

vector<Mat> colorTransfer(vector<Mat>& inputplane, vector<Mat>& modelplane) {
	vector<Mat> resultplane;
	Mat lMeanIn, lStdIn, aMeanIn, aStdIn, bMeanIn, bStdIn;
	Mat lMeanMo, lStdMo, aMeanMo, aStdMo, bMeanMo, bStdMo;
	Mat l_plane = inputplane[0];
	Mat a_plane = inputplane[1];
	Mat b_plane = inputplane[2];

	meanStdDev(inputplane[0], lMeanIn, lStdIn);
	meanStdDev(inputplane[1], aMeanIn, aStdIn);
	meanStdDev(inputplane[2], bMeanIn, bStdIn);
	meanStdDev(modelplane[0], lMeanMo, lStdMo);
	meanStdDev(modelplane[1], aMeanMo, aStdMo);
	meanStdDev(modelplane[2], bMeanMo, bStdMo);
	
	l_plane.convertTo(l_plane, CV_64F);
	a_plane.convertTo(a_plane, CV_64F);
	b_plane.convertTo(b_plane, CV_64F);
	
	//subtract the means from the model
	l_plane = l_plane - lMeanIn;
	a_plane = a_plane - aMeanIn;
	b_plane = b_plane - bMeanIn;
	//scale by the standard deviations
	l_plane = (lStdMo / lStdIn).mul(l_plane);
	a_plane = (aStdMo / aStdIn).mul(a_plane);
	b_plane = (bStdMo / bStdIn).mul(b_plane);
	//add in the input mean
	l_plane = l_plane + lMeanMo;
	a_plane = a_plane + aMeanMo;
	b_plane = b_plane + bMeanMo;
	
	l_plane.convertTo(l_plane, CV_8U);
	a_plane.convertTo(a_plane, CV_8U);
	b_plane.convertTo(b_plane, CV_8U);
	
	resultplane.push_back(l_plane);
	resultplane.push_back(a_plane);
	resultplane.push_back(b_plane);

	return resultplane;
}