/*
LoadModel loads a SavedModel file and produces a prediction for a given input tensor. This prediction is written to output.csv
*/

#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NeuralNet.h"
#include "ConsoleOutput.h"
#include <cstdlib>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

using namespace tensorflow;

int main(){
	//this sets the directory containing the required model and .csv files, needs to be adjusted accordingly
	std::string folder=std::getenv("HOME")+std::string("/local/tensorflow_cc-master/test/");
	
	//loads config from csv file, contains learning rate, number of epochs (both are only used by Standalone.cpp) and the name of the model to be loaded
	const auto config=getConfigFromCSV(folder+"config.csv");
	
	//loads input and label data from given .csv files into tensors
	//the first two integers in SetupBatchesExcludingGhost denote the columns of the csv that contain the x component of the velocity
	//the last integer is the batch size, only relevant for training
	std::vector<tensorflow::Tensor> input, label;
	std::string addendum="_clean";
	SetupBatchesExcludingGhost(input, folder+"writer2"+addendum+".csv", 8, label, folder+"writer1"+addendum+".csv", 4, 1);
	
	//loads SavedModel file into "model"
	SavedModelBundleLite model;
	SessionOptions session_options = SessionOptions();
	RunOptions run_options = RunOptions();
	Status status = LoadSavedModel(session_options, run_options, "./"+config[2], {kSavedModelTagServe}, &model);	
	//checks if SavedModel was loaded correctly
	if (!status.ok()) {
		std::cerr << "Failed: " << status;
		return 3;
	}
	
	
	std::vector<Tensor> results;
    float loss;
	std::vector<float> res;
	std::vector<std::vector<float>> csvWrite;
	std::vector<Tensor> out_tensors;
	
	//selects time step for prediction and comparison
	const int compTime=249;
	
	//gets names of input and output of the model to later address them
	//names can also be accessed from console with: saved_model_cli show --dir ~/local/tensorflow_cc-master/test/model --all
	std::string input_name = model.GetSignatures().at("serving_default").inputs().begin()->second.name();
	std::string output_name=model.GetSignatures().at("serving_default").outputs().begin()->second.name();
	
	//computes network predictions
	//loop is only really useful for prob model, since its weights are sampled each time
	for(int i=0; i<100; i++){
		//executes model for a given input and writes prediction to "results"
		Status runStatus = model.GetSession()->Run({{input_name, input[compTime]}}, {output_name}, {}, &results);
		std::cout<<runStatus<<std::endl;
		
		//prints prediction and the difference to the label data
		res=TensorToVec(results[0])[0];
		std::cout<<res<<std::endl;
		std::vector<float> AB = res;
		res-=TensorToVec(label[compTime])[0];
		float avg=0;
		for(float number : res) avg+=std::abs(number);
		std::cout<<avg/res.size()<<std::endl;
		AB.insert(AB.end(), res.begin(), res.end());
		csvWrite.push_back(AB);
	}
	
	//writes prediction and comparison to output.csv
	VecToCSV(csvWrite);
	
	return 1;
}

