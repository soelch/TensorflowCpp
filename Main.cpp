#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NN.h"
#include "ConsoleOutput.h"

using namespace tensorflow;

int main(){
	tensorflow::Tensor input=getCSVasTensor("test.csv");
	tensorflow::Tensor label=getCSVasTensor("test2.csv");
	std::cout<<"hi"<<std::endl;
	Scope root = Scope::NewRootScope();
	
	
	
	std::vector<std::vector<float>> results;
    float loss;
	std::vector<float> res;
	NeuralNet NN(3, 3, 1);
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph(0.0005f);
	NN.Initialize();

	tensorflow::Tensor temp(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,3}));
	temp.flat<float>()(0) = 0.2f;
	temp.flat<float>()(1) = 0.4f;
	temp.flat<float>()(2) = 0.8f;
	std::cout<<temp.DebugString()<<std::endl;

	for(int i=0; i<25000; ++i) NN.TrainNN(input, label, results, loss), std::cout<<loss<<std::endl;
	NN.Predict(temp, res);
	std::cout<<res<<std::endl;

	return 1;
}
