#pragma once

#include <vector>
#include <iostream>


std::ostream& operator<< (std::ostream& s, const std::vector<double> vec){
	for(auto entry : vec) s<<entry<<"  ";
	return s;
}

std::ostream& operator<< (std::ostream& s, const std::vector<std::vector<double>> vec){
	for(auto entry : vec) s<<entry<<"\n"<<std::endl;
	return s;
}
