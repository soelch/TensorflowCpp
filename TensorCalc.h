#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "ConsoleOutput.h"
#include "ReadCSV.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

float tensorMean(tensorflow::Tensor t){
	const int max=t.dim_size(0);
	float sum=0;
	for(int i=0;i<max;++i){
		sum+=t.flat<float>()(i);
	}
	return sum/(float)max;
}

std::vector<float> operator-=(std::vector<float>& vec1, std::vector<float> vec2){
	if(vec1.size()!=vec2.size()){ std::cout<<"\nVectors are not the same size"<<std::endl; return std::vector<float>(1,-1.0f);}
	
	for(int i=0; i<vec1.size();++i){
		vec1[i]-=vec2[i];
	}
	return vec1;
}












