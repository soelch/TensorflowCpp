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
	SetupBatches(input, folder+"writer2.csv", 8, label, folder+"CleanTraining250.csv", 4, 3);


	Scope root = Scope::NewRootScope();
	
	std::vector<std::vector<float>> results;
    float loss;
	std::vector<float> res;
	NeuralNet NN(input[0].dim_size(1), input[0].dim_size(1), label[0].dim_size(1));
	NN.CreateGraphForNN();
	NN.CreateOptimizationGraph((float)config[0][0]);
	NN.Initialize();
	
	float total=0.0, previous, trainRate=(float)config[0][0];
	int ctr=0, limit=2;
	const int nBatches=input.size();
	for(int i=0; i<config[0][1]; i++){ 
		for(int j=0; j<nBatches; j++){  //<---does not necessarily start at 0!!!!!!!!
			NN.TrainNN(input[j], label[j], results, loss); 
			total+=loss;
		}

		total=total/(float)nBatches;
		if(std::abs(previous-total)<total/200) ctr++;
		else ctr=0;

		if(ctr>limit) limit++, trainRate*=0.75, NN.UpdateOptimizationGraph(trainRate),ctr=0;
		std::cout<<i<<" "<<trainRate<<" "<<total<<" "<<ctr<<std::endl; 
		previous=total;
		total=0.0;
	}
	

	NN.Predict(getTensorByIndex(input[65],1), res);
	std::cout<<res<<std::endl;
	std::vector<float> AB = res;
	res-=TensorToVec(getTensorByIndex(label[65],1))[0];
	float avg=0;
	for(float number : res) avg+=std::abs(number);
	std::cout<<avg/res.size()<<std::endl;
	AB.insert(AB.end(), res.begin(), res.end());
	VecToCSV(AB);
	
	return 1;
}
