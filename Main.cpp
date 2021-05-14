#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NN.h"

using namespace tensorflow;

int main(){
	tensorflow::Tensor input=getCSVasTensor("writer1_200.csv", 7);
	tensorflow::Tensor label=getCSVasTensor("writer2_200.csv", 7);
	Scope root = Scope::NewRootScope();
	
/*
	auto a = Const(root, input);
	auto b = Const(root, { {2, 2}, {1, 1} });
	auto m = ExpandDims(root, input, 0);
	ClientSession session(root);
	std::vector<Tensor> outputs;
	session.Run({m}, &outputs);
	std::cout<<outputs[0].DebugString()<<std::endl;
*/
	vector<float> results;
    float loss;
	NeuralNet NN(3);
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph(0.0005f);
	NN.Initialize();
	std::cout<<loss<<std::endl;
	tensorflow::Tensor temp(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,3}));
	temp.flat<float>()(0) = 0.2f;
	temp.flat<float>()(1) = 0.4f;
	temp.flat<float>()(2) = 0.8f;
	std::cout<<temp.DebugString()<<std::endl;
	NN.Predict(temp, loss);
	std::cout<<loss<<std::endl;
	NN.TrainNN(input, label, results, loss);
	std::cout<<results[0]<<std::endl;
	NN.Predict(temp, loss);
	std::cout<<loss<<std::endl;
	for(int i=0; i<50000; ++i) NN.TrainNN(input, label, results, loss), std::cout<<loss<<std::endl;
	NN.Predict(temp, loss);
	std::cout<<loss<<std::endl;

/*
	vector<Tensor> out_tensors;
	std::cout<<input.DebugString()<<std::endl;
	auto in=Placeholder(root.WithOpName("input"), DT_FLOAT);
	auto dims=ExpandDims(root.WithOpName("dims"), in, 0);
	TF_CHECK_OK(session.Run({{in , input}},{dims}, &out_tensors));
	std::cout<<out_tensors[0].DebugString()<<std::endl;
*/
	return 1;
}
