
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

#include <iostream>
#include <map>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/cc/framework/scope.h"
#include "TensorCalc.h"

using namespace tensorflow;
using namespace tensorflow::ops;

auto tf_tensor_to_vector(tensorflow::Tensor tensor, int32_t tensorSize);

class NeuralNet
{
private:
	int input_size, middle_size, output_size;
    Scope i_root;
    Output file_name_var;
    Output image_tensor_var;
    Scope a_root;
    Scope net_scope;
    std::unique_ptr<ClientSession> t_session;
    std::unique_ptr<Session> f_session;
    Output input_placeholder;
    string input_name = "input";
    Output label_placeholder;
    Output out_classification;
    string out_name = "output_classes";
    //Network maps
    std::map<string, Output> m_vars;
    std::map<string, TensorShape> m_shapes;
    std::map<string, Output> m_assigns;
    //Loss variables
    std::vector<Output> v_weights_biases;
	std::vector<Output> grad_outputs;
    std::vector<Operation> v_out_grads;
    Output out_loss_var;
    //InputList MakeTransforms(int batch_size, Input a0, Input a1, Input a2, Input b0, Input b1, Input b2);

public:
	NeuralNet():input_size(0), middle_size(0), output_size(0), i_root(Scope::NewRootScope()), a_root(Scope::NewRootScope()), net_scope(Scope::NewRootScope()){}
    NeuralNet(int in, int middle, int out): input_size(in), middle_size(middle), output_size(out),  i_root(Scope::NewRootScope()), a_root(Scope::NewRootScope()), net_scope(Scope::NewRootScope()) {} 
	void CreateNN(int in, int middle, int out);
    Input Init(Scope scope, int in_chan, int out_chan);
    Input AddDenseLayer(string idx, Scope scope, int in_units, int out_units, bool bActivation, Input input);
    Output AddOutLayer(Scope scope, int in_units, int out_units, Input input);
    Status CreateNNGraph();
    Status CreateOptimizationGraph(float learning_rate);
	Status UpdateOptimizationGraph(float learning_rate);
    Status Initialize();
    Status Train(Tensor& image_batch, Tensor& label_batch, std::vector<std::vector<float>>& results, float& loss);
    Status ValidateNN(Tensor& image_batch, Tensor& label_batch, std::vector<float>& results);
    Status Predict(Tensor image, std::vector<float>& result);
};

