#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NN.h"

using namespace tensorflow;

int main(){
	tensorflow::Tensor input=(getCSVasTensor("test.csv"));
	tensorflow::Tensor label=(getCSVasTensor("test2.csv"));
	std::cout<<input.DebugString(100)<<"\n";
	Scope root = Scope::NewRootScope();
	// 2x2 matrix
	auto a = Const(root, input);
	// 2x2 matrix
	auto b = Const(root, { {2, 2}, {1, 1} });
	// a x b
	auto m = ExpandDims(root, a, 0);
	ClientSession session(root);
	std::vector<Tensor> outputs;
	session.Run({m}, &outputs);
	std::cout<<outputs[0].DebugString()<<std::endl;

	vector<float> results;
        float loss;
	NeuralNet NN(1);
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph(0.005f);
	NN.Initialize();
	std::cout<<loss<<std::endl;
	tensorflow::Tensor temp(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,1}));
	temp.scalar<float>()() = 1.0f;
	std::cout<<temp.DebugString()<<std::endl;
	//tensorflow::Input::Initializer temp({1}, tensorflow::TensorShape({1}));
	NN.Predict(temp, loss);
	std::cout<<loss<<std::endl;
	NN.TrainNN(input, label, results, loss);
	std::cout<<results[0]<<std::endl;
	NN.Predict(temp, loss);
	std::cout<<loss<<std::endl;
	for(int i=0; i<50; ++i) NN.TrainNN(input, label, results, loss), std::cout<<loss<<std::endl;
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
