/*
Copied from https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c?page=1&tab=votes#tab-top
*/
#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "ConsoleOutput.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ';'))
    {
        result.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

class CSVRow
{
    public:
        std::string_view operator[](std::size_t index) const
        {
            return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
        }
        std::size_t size() const
        {
            return m_data.size() - 1;
        }
        void readNextRow(std::istream& str)
        {
            std::getline(str, m_line);

            m_data.clear();
            m_data.emplace_back(-1);
            std::string::size_type pos = 0;
            while((pos = m_line.find(';', pos)) != std::string::npos)
            {
                m_data.emplace_back(pos);
                ++pos;
            }
            // This checks for a trailing comma with no data after it.
            pos   = m_line.size();
            m_data.emplace_back(pos);
        }
    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}   

class CSVIterator
{   
    public:
        typedef std::input_iterator_tag     iterator_category;
        typedef CSVRow                      value_type;
        typedef std::size_t                 difference_type;
        typedef CSVRow*                     pointer;
        typedef CSVRow&                     reference;

        CSVIterator(std::istream& str)  :m_str(str.good()?&str:NULL) { ++(*this); }
        CSVIterator()                   :m_str(NULL) {}

        // Pre Increment
        CSVIterator& operator++()               {if (m_str) { if (!((*m_str) >> m_row)){m_str = NULL;}}return *this;}
        // Post increment
        CSVIterator operator++(int)             {CSVIterator    tmp(*this);++(*this);return tmp;}
        CSVRow const& operator*()   const       {return m_row;}
        CSVRow const* operator->()  const       {return &m_row;}

        bool operator==(CSVIterator const& rhs) {return ((this == &rhs) || ((this->m_str == NULL) && (rhs.m_str == NULL)));}
        bool operator!=(CSVIterator const& rhs) {return !((*this) == rhs);}
    private:
        std::istream*       m_str;
        CSVRow              m_row;
};

//turns 2d vector into 2d tensor with the same entries
//all 1d vectors must be the same length 
const tensorflow::Tensor VecToTensor(std::vector<std::vector<float>> vec){
	
	tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)vec.size(), (int)vec[0].size()}));
	auto input_map = input.tensor<float, 2>();
	
	for(int i=0;i<vec.size();++i){
		for(int j=0;j<(int)vec[i].size();++j){
			input_map(i, j) = vec[i][j];
		}
	}
	return input;
}

float maxInColumn(std::vector<std::vector<float>> vec, int column){
	float max = vec[0][column];
	for (int i=1;i<vec.size();++i){
		if(vec[i][column]>max) max=vec[i][column];
	}
	return max;
}

const tensorflow::Tensor VecToTensor(std::vector<float> vec){
	
	tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,(int)vec.size()}));
	auto input_map = input.tensor<float, 2>();
	
	for(int j=0;j<(int)vec.size();++j){
		input_map(0, j) = vec[j];
	}
	return input;
}

std::vector<std::vector<float>> TensorToVec(tensorflow::Tensor t){
	std::vector<std::vector<float>> vec(t.dim_size(0),std::vector<float>(t.dim_size(1), -1));
	auto input_map = t.tensor<float, 2>();
	for(int i=0;i<(int)t.dim_size(0);++i){
		for(int j=0;j<(int)t.dim_size(1);++j){
			 vec[i][j]=input_map(i, j);
		}
	}
	return vec;
}

std::vector<std::vector<float>> getCSVasVec(std::string filename){
	std::ifstream file(filename);
	CSVIterator loop(file);
	std::vector<std::vector<float>> vec;
	std::vector<float> temp;

    for(loop; loop != CSVIterator(); ++loop){
		temp.clear();
		//this avoids empty lines
		if((*loop)[0]==""||(*loop)[0]==" ") continue;
		for(int i=0;i<(*loop).size();++i){
			if((*loop)[i]!=""&&(*loop)[i]!=" ")temp.push_back((float)std::stod(std::string((*loop)[i])));
		}
		vec.push_back(temp);
	}
	return vec;
}

std::vector<std::vector<float>> getCSVasVec(std::vector<std::string> filenameVec){
	std::vector<std::vector<float>> vec, temp;

    	for(auto filename : filenameVec){
		temp=getCSVasVec(filename);
		vec.insert(vec.end(), temp.begin(), temp.end());
	}
	
	return vec;
}

//returns 2d vector containing the [index]-entry of every line of the csv
//first vector index denotes the timestep-1, second index denotes macro coordinate(from 4,4,4 over 5,4,4 to 10,10,10)
std::vector<std::vector<float>> getCSVasVec(std::string filename, int index){
	std::vector<std::vector<float>> temp=getCSVasVec(filename);
	std::vector<std::vector<float>> vec(maxInColumn(temp, 0),std::vector<float>());
	for(auto element : temp) vec[element[0]-1].push_back(element[index]);
	return vec;
}

std::vector<float> getCSVEntryasVec(std::string filename, int index){
	std::ifstream file(filename);
	CSVIterator loop(file);
	std::vector<float> vec;
	
    for(loop; loop != CSVIterator(); ++loop){
		vec.push_back((float)(std::stod(std::string((*loop)[index]))));
	}
	
	return vec;
}

std::vector<std::vector<float>> getCSVEntryasVec(std::vector<std::string> filenameVec, int index){
	std::vector<std::vector<float>> vec;
	for(auto filename : filenameVec){
		vec.push_back(getCSVEntryasVec(filename, index));
	}
	return vec;
}

const tensorflow::Tensor getCSVasTensor(std::string filename){
	return VecToTensor(getCSVasVec(filename));
}

const tensorflow::Tensor getCSVasTensor(std::vector<std::string> filenameVec){
	return VecToTensor(getCSVasVec(filenameVec));
}

const tensorflow::Tensor getCSVEntriesasTensor(std::vector<std::string> filenameVec, int index){
	return VecToTensor(getCSVEntryasVec(filenameVec, index));
}

const tensorflow::Tensor getCSVasTensor(std::string filename, int index){
	return VecToTensor(getCSVasVec(filename, index));
}

tensorflow::Tensor getTensorByIndex(tensorflow::Tensor t, int index){
	return VecToTensor(TensorToVec(t)[index]);
}







