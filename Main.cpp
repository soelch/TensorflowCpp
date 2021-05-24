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
	std::vector<tensorflow::Tensor> input, label;
	SetupBatches(input, folder+"writer2_200.csv", 4, label, folder+"writer1_200.csv", 7);
	
	Scope root = Scope::NewRootScope();
	
	std::vector<std::vector<float>> results;
    float loss;
	std::vector<float> res;
	NeuralNet NN(input[0].dim_size(1), input[0].dim_size(1), label[0].dim_size(1));
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph((float)config[0][0]);
	NN.Initialize();
	
	float total=0.0, previous, trainRate=(float)config[0][0];
	int ctr=0;
	const int nBatches=input.size();
	for(int i=0; i<config[0][1]; i++){ 
		for(int j=0; j<nBatches; j++){
			NN.TrainNN(input[j], label[j], results, loss); 
			total+=loss;
		}
		std::cout<<total<<std::endl;
		total=total/(float)nBatches;
		if(std::abs(previous-total)<total/100) ctr++;
		else ctr=0;
		//if(ctr>3) trainRate*=1.5, NN.ChangeLearningRate(trainRate),ctr=0;
		std::cout<<i<<" "<<trainRate<<" "<<total<<std::endl; 
		previous=total;
		total=0.0;
	}
	

	NN.Predict(input[249], res);
	std::cout<<res<<std::endl;
	std::vector<float> AB = res;
	res-=TensorToVec(label[249])[0];
	float avg=0;
	for(float number : res) avg+=std::abs(number);
	std::cout<<avg/res.size()<<std::endl;
	AB.insert(AB.end(), res.begin(), res.end());
	VecToCSV(AB);
	
	return 1;
}
