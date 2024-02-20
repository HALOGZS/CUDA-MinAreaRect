#pragma once
#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<opencv2/opencv.hpp>


void MinAreaRectF(std::vector<std::vector<cv::Point>>contours, std::vector<std::vector<float>>& result, cudaStream_t stream = 0);



