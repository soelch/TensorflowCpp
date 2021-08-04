/*
Standalone creates a TensorFlow graph with the help of NeuralNet.h and trains it with a given set of input and label data
*/

#include "ReadCSV.h"
#include <vector>
#include <unordered_map>
#include "tensorflow/core/framework/tensor.h"
#include "NeuralNet.h"
#include "ConsoleOutput.h"
#include <cstdlib>

using namespace tensorflow;

int main(){
	//this sets the directory containing the required .csv files, needs to be adjusted accordingly
	std::string folder=std::getenv("HOME")+std::string("/local/tensorflow_cc-master/test/");
	
	//loads config from csv file, contains learning rate and number of epochs
	const auto config=getCSVasVec(folder+"config_main.csv");
	
	//loads input and label data from given .csv files into tensors
	//the first two integers in SetupBatchesExcludingGhost denote the columns of the csv that contain the x component of the velocity
	//the last integer is the batch size used for training
	std::vector<tensorflow::Tensor> input, label;
	std::string addendum="_clean";
	SetupBatchesExcludingGhost(input, folder+"writer2"+addendum+".csv", 8, label, folder+"writer1"+addendum+".csv", 4, 1);

	
	std::vector<std::vector<float>> results;
    float loss;
	std::vector<float> res;
	
	//initializes NeuralNet graph with given dimensions and learning rate
	NeuralNet NN(input[0].dim_size(1), input[0].dim_size(1), label[0].dim_size(1));
	NN.CreateNNGraph();
	NN.CreateOptimizationGraph((float)config[0][0]);
	NN.Initialize();
	
	float total=0.0, previous, trainRate=(float)config[0][0];
	int ctr=0, limit=2;
	const int nBatches=input.size();

	//trains the network for a given number of epochs and prints the loss each time
	for(int i=0; i<config[0][1]; i++){ 
		for(int j=0; j<nBatches; j++){
			NN.Train(input[j], label[j], results, loss); 
			total+=loss;
		}
		total=total/(float)nBatches;
		std::cout<<"Loss: "<<total<<std::endl;
		/*
		//this block checks if the network loss changed by less than a given threshold between 
		//the last and the current iteration and increases a counter or resets it accordingly
		//if the counter is larger than "limit", the learning rate is adjusted
		//this feature is not fully tested and may not work as intended
		if(std::abs(previous-total)<total/200) ctr++;
		else ctr=0;
		previous=total;
		if(ctr>limit) limit++, trainRate*=0.75, NN.UpdateOptimizationGraph(trainRate),ctr=0;
		*/
		total=0.0;
	}
	
	//computes a prediction for a given timestep
	//for batch sizes other than 1, the index for "input" is not equal to the time step
	NN.Predict(getTensorByIndex(input[65],0), res);
	std::cout<<res<<std::endl;
	std::vector<float> AB = res;
	std::cout<<res.size();
	std::cout<<getTensorByIndex(label[65],0).DebugString()<<std::endl;
	std::cout<<TensorToVec(getTensorByIndex(label[65],0))[0].size();
	res-=TensorToVec(getTensorByIndex(label[65],0))[0];
	AB.insert(AB.end(), res.begin(), res.end());
	
	//writes prediction and absolute error to output.csv
	VecToCSV(AB);
	
	return 1;
}
