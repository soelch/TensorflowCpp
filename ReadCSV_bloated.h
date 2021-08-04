/*
Contains methods/classes used for interaction with csv files
*/
#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "ConsoleOutput.h"
#include "TensorCalc.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <string_view>

//holds current row of csv file and positions of all entries within the row in order to access them individually
//only looks for ";" not ","
class CSVRow
{
    public:
		//gets one element from current row
        std::string_view operator[](std::size_t index) const;
		
		//returns size, i.e. number of semicolons in row
        std::size_t size() const;
		
		//updates  m_line to next line in csv file and saves positions of semicolons in m_data
        void readNextRow(std::istream& str);
    private:
        std::string         m_line;
        std::vector<int>    m_data;
};

//shorthand for readNextRow
std::istream& operator>>(std::istream& str, CSVRow& data);

//holds CSVRow, used to iterate through csv file
class CSVIterator
{   
    public:
        typedef std::input_iterator_tag     iterator_category;
        typedef CSVRow                      value_type;
        typedef std::size_t                 difference_type;
        typedef CSVRow*                     pointer;
        typedef CSVRow&                     reference;
		
		//this also increments once to get first row in csv 
        CSVIterator(std::istream& str);
        CSVIterator();

        //increment csv row
        CSVIterator& operator++();
		
		//get m_row
        CSVRow const& operator*()   const;
		
		//get reference to m_row
        CSVRow const* operator->()  const;
		
		//check if equal
        bool operator==(CSVIterator const& rhs);
		
		//check if  not equal
        bool operator!=(CSVIterator const& rhs);
    private:
        std::istream*       m_str;
        CSVRow              m_row;
};

//gets elements of first rwo of csv as strings
std::vector<std::string> getConfigFromCSV(std::string filename);

//gets csv as 2D vec of floats
//first dimension denotes row, second denotes element in row
//this and the function after it should be the only ones that do not sort csv/vector by time step
std::vector<std::vector<float>> getCSVasVec(std::string filename);

//gets multiple csv as 2D vec of floats
std::vector<std::vector<float>> getCSVasVec(std::vector<std::string> filenameVec);

//returns 2d vector containing the [index]-entry of every line of the csv
//first vector index is sorted by timestep, second index denotes macro coordinate(from 4,4,4 over 5,4,4 to 10,10,10)
std::vector<std::vector<float>> getCSVasVec(std::string filename, int index);

//same as above, but each entry is divided by mass which is indicated by the divisor index
std::vector<std::vector<float>> getCSVasVec(std::string filename, int index, int divisor);

//same as getCSVasVec(std::string filename, int index) but excludes ghost cells, i.e. index 0 and 13
std::vector<std::vector<float>> getCSVasVecExcludingGhost(std::string filename, int index);

//same as getCSVasVec(std::string filename, int index, int divisor) but excludes ghost cells, i.e. index 0 and 13
std::vector<std::vector<float>> getCSVasVecExcludingGhost(std::string filename, int index, int divisor);

//gets single column of csv as 1D vec of floats
std::vector<float> getCSVEntryasVec(std::string filename, int index);

//gets single column of multiple csv as 1D vec of floats
std::vector<std::vector<float>> getCSVEntryasVec(std::vector<std::string> filenameVec, int index);

//gets csv as 2D Tensor
const tensorflow::Tensor getCSVasTensor(std::string filename);

//gets multiple csv as 2D Tensor
const tensorflow::Tensor getCSVasTensor(std::vector<std::string> filenameVec);

//gets one column from multiple csv as Tensor
const tensorflow::Tensor getCSVEntriesasTensor(std::vector<std::string> filenameVec, int index);

//gets one column from csv as Tensor
const tensorflow::Tensor getCSVasTensor(std::string filename, int index);

//gets one column from multiple csv as Tensor and divides it by the divisor column
const tensorflow::Tensor getCSVasTensor(std::string filename, int index, int divisor);

//gets one column from multiple csv as vector of Tensors, makes it easier to access individual tensors
const std::vector<tensorflow::Tensor> getCSVasVecOfTensors(std::string filename, int index);

const std::vector<tensorflow::Tensor> getCSVasVecOfTensors(std::string filename, int index, int divisor);

const std::vector<tensorflow::Tensor> getCSVasVecOfBatches(std::string filename, int index, int batch_size);

const std::vector<tensorflow::Tensor> getCSVasVecOfBatches(std::string filename, int index, int batch_size, int divisor);

const std::vector<tensorflow::Tensor> getCSVasVecOfBatchesExcludingGhost(std::string filename, int index, int batch_size);

const std::vector<tensorflow::Tensor> getCSVasVecOfBatchesExcludingGhost(std::string filename, int index, int batch_size, int divisor);

void VecToCSV(std::vector<std::vector<float>> vec2d);

void VecToCSV(std::vector<float> vec);

void SetupTensors(tensorflow::Tensor& in, std::string in_path, int in_column, tensorflow::Tensor& label, std::string label_path, int label_column);

void SetupBatches(std::vector<tensorflow::Tensor>& in, std::string in_path, int in_column, std::vector<tensorflow::Tensor>& label, std::string label_path, int label_column, int batch_size);

void SetupBatches(std::vector<tensorflow::Tensor>& in, std::string in_path, int in_column, std::vector<tensorflow::Tensor>& label, std::string label_path, int label_column, int batch_size, int divisor);

void SetupBatchesExcludingGhost(std::vector<tensorflow::Tensor>& in, std::string in_path, int in_column, std::vector<tensorflow::Tensor>& label, std::string label_path, int label_column, int batch_size);

void SetupBatchesExcludingGhost(std::vector<tensorflow::Tensor>& in, std::string in_path, int in_column, std::vector<tensorflow::Tensor>& label, std::string label_path, int label_column, int batch_size, int divisor);