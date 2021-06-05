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
	std::string folder=std::getenv("HOME")+std::string("/local/tensorflow_cc-master/test/");
	const auto config=getCSVasVec(folder+"config.csv");
	std::cout<<config<<std::endl;
	std::vector<tensorflow::Tensor> input, label;
	std::string addendum="_clean";
	SetupBatchesExcludingGhost(input, folder+"writer2"+addendum+".csv", 8, label, folder+"writer1"+addendum+".csv", 4, 1);
	
	SavedModelBundleLite model;
	SessionOptions session_options = SessionOptions();
	RunOptions run_options = RunOptions();
	Status status = LoadSavedModel(session_options, run_options, "./model", {kSavedModelTagServe}, &model);	
	if (!status.ok()) {
		std::cerr << "Failed: " << status;
		return 3;
	}
	
	std::vector<Tensor> results;
    float loss;
	std::vector<float> res;
	
	
	//get names with      saved_model_cli show --dir ~/local/tensorflow_cc-master/test/model --all
	//serving_default_input appears to increment with each new model
	std::string input_name = model.GetSignatures().at("serving_default").inputs().begin()->second.name();
	std::string output_name=model.GetSignatures().at("serving_default").outputs().begin()->second.name();
	//std::vector<std::string> output_name{"StatefulPartitionedCall:0"};
	std::vector<Tensor> out_tensors;
	Status runStatus = model.GetSession()->Run({{input_name, input[100]}}, {output_name}, {}, &results);
	std::cout<<runStatus<<std::endl;
	
	
	
	res=TensorToVec(results[0])[0];
	std::cout<<res<<std::endl;
	std::vector<float> AB = res;
	res-=TensorToVec(label[65])[0];
	float avg=0;
	for(float number : res) avg+=std::abs(number);
	std::cout<<avg/res.size()<<std::endl;
	AB.insert(AB.end(), res.begin(), res.end());
	VecToCSV(AB);
	
	return 1;
}

