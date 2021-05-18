#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NeuralNet.h"
#include "ConsoleOutput.h"
#include <cstdlib>

using namespace tensorflow;

int main(){
	std::string folder=std::getenv("HOME")+std::string("/local/tensorflow_cc-master/test/");
	const auto config=getCSVasVec(folder+"config.csv");
	std::cout<<config<<std::endl;
	tensorflow::Tensor input=getCSVasTensor(folder+"writer2_200.csv", 4);
	tensorflow::Tensor label=getCSVasTensor(folder+"writer1_200.csv", 7);
	
	Scope root = Scope::NewRootScope();
	
	std::vector<std::vector<float>> results;
    float loss;
	std::vector<float> res;
	NeuralNet NN(input.dim_size(1), input.dim_size(1), label.dim_size(1));
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph((float)config[0][0]);
	NN.Initialize();

	tensorflow::Tensor temp(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,3}));
	temp.flat<float>()(0) = 0.2f;
	temp.flat<float>()(1) = 0.4f;
	temp.flat<float>()(2) = 0.8f;
	std::cout<<temp.DebugString()<<std::endl;

	for(int i=0; i<config[0][1]; ++i){ 
	NN.TrainNN(input, label, results, loss); 
	std::cout<<loss<<std::endl; 
	if(i%100==0) std::cout<<"\n"<<i<<"\n"<<std::endl;
	}
	
	auto pred=getTensorByIndex(input,249);
	NN.Predict(pred, res);
	std::cout<<res<<std::endl;
	
	if(std::remove("output.csv")==0) std::cout<<"output.csv was deleted"<<std::endl;
	res-=TensorToVec(label)[249];
	
	std::ofstream myfile("output.csv");
    
    for(float number : res)
    {
		myfile << number << '\t';
		myfile << "," ;
    }
	
	return 1;
}
