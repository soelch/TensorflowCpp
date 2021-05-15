#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NN.h"
#include "ConsoleOutput.h"

using namespace tensorflow;

int main(){
	tensorflow::Tensor input=getCSVasTensor("writer1_200.csv", 7);
	tensorflow::Tensor label=getCSVasTensor("writer2_200.csv", 4);
	
	Scope root = Scope::NewRootScope();
	
	std::vector<std::vector<float>> results;
    float loss;
	std::vector<float> res;
	NeuralNet NN(input.dim_size(1), input.dim_size(1), label.dim_size(1));
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph(0.001f);
	NN.Initialize();

	tensorflow::Tensor temp(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,3}));
	temp.flat<float>()(0) = 0.2f;
	temp.flat<float>()(1) = 0.4f;
	temp.flat<float>()(2) = 0.8f;
	std::cout<<temp.DebugString()<<std::endl;

	for(int i=0; i<1000; ++i){ 
	NN.TrainNN(input, label, results, loss); 
	std::cout<<loss<<std::endl; 
	if(i%100==0) std::cout<<"\n"<<i<<"\n"<<std::endl;
	}
	
	auto pred=getTensorByIndex(input,249);
	NN.Predict(pred, res);
	std::cout<<res<<std::endl;
	
	res-=TensorToVec(label)[249];
	
	ofstream myfile("output.csv");
    
    for(float number : res)
    {
		myfile << number << '\t';
		myfile << "," ;
    }
	
	return 1;
}
