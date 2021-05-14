
#include "tensorflow/cc/ops/array_ops.h"
#include <tensorflow/cc/ops/standard_ops.h>
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include <tensorflow/cc/client/client_session.h>

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
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
class NeuralNet
{
private:
    Scope i_root; //graph for loading images into tensors
    const int input_size;
    //load image vars
    Output file_name_var;
    Output image_tensor_var;
    //data augmentation
    Scope a_root;
    Output aug_tensor_input;
    Output aug_tensor_output;
    //training and validating the CNN
    Scope t_root; //graph
    unique_ptr<ClientSession> t_session;
    unique_ptr<Session> f_session;
    //CNN vars
    Output input_batch_var;
    string input_name = "input";
    Output input_labels_var;
    Output drop_rate_var; //use real drop rate in training and 1 in validating
    string drop_rate_name = "drop_rate";
    Output skip_drop_var; //use 0 in trainig and 1 in validating
    string skip_drop_name = "skip_drop";
    Output out_classification;
    string out_name = "output_classes";
    Output logits;
    //Network maps
    map<string, Output> m_vars;
    map<string, TensorShape> m_shapes;
    map<string, Output> m_assigns;
    //Loss variables
    vector<Output> v_weights_biases;
    vector<Operation> v_out_grads;
    Output out_loss_var;
    InputList MakeTransforms(int batch_size, Input a0, Input a1, Input a2, Input b0, Input b1, Input b2);
public:
    NeuralNet(int in):i_root(Scope::NewRootScope()), t_root(Scope::NewRootScope()), a_root(Scope::NewRootScope()),input_size(in) {} 
    Input XavierInit(Scope scope, int in_chan, int out_chan);
    Input AddDenseLayer(string idx, Scope scope, int in_units, int out_units, bool bActivation, Input input);
    Status CreateGraphForNN();
    Status CreateOptimizationGraph(float learning_rate);
    Status Initialize();
    Status TrainNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results, float& loss);
    Status ValidateNN(Tensor& image_batch, Tensor& label_batch, vector<float>& results);
    Status Predict(Tensor& image, int& result);
    Status FreezeSave(string& file_name);
    Status LoadSavedModel(string& file_name);
    Status PredictFromFrozen(Tensor& image, int& result);
    Status CreateAugmentGraph(int batch_size, int image_side, float flip_chances, float max_angles, float sscale_shift_factor);
    Status RandomAugmentBatch(Tensor& image_batch, Tensor& augmented_batch);
    Status WriteBatchToImageFiles(Tensor& image_batch, string folder_name, string image_name);
};

Input NeuralNet::XavierInit(Scope scope, int in_chan, int out_chan)
{
    float std;
    Tensor t;
        std = sqrt(6.f/(in_chan+out_chan));
        Tensor ts(DT_INT64, {2});
        auto v = ts.vec<int64>();
        v(0) = in_chan;
        v(1) = out_chan;
        t = ts;
    auto rand = RandomUniform(scope, t, DT_FLOAT);
    return Multiply(scope, Sub(scope, rand, 0.5f), std*2.f);
}


Input NeuralNet::AddDenseLayer(string idx, Scope scope, int in_units, int out_units, bool bActivation, Input input)
{
    TensorShape sp = {in_units, out_units};
    m_vars["W"+idx] = Variable(scope.WithOpName("W"), sp, DT_FLOAT);
    m_shapes["W"+idx] = sp;
    m_assigns["W"+idx+"_assign"] = Assign(scope.WithOpName("W_assign"), m_vars["W"+idx], XavierInit(scope, in_units, out_units));
    sp = {out_units};
    m_vars["B"+idx] = Variable(scope.WithOpName("B"), sp, DT_FLOAT);
    m_shapes["B"+idx] = sp;
    m_assigns["B"+idx+"_assign"] = Assign(scope.WithOpName("B_assign"), m_vars["B"+idx], Input::Initializer(0.f, sp));
    auto dense = Add(scope.WithOpName("Dense_b"), MatMul(scope.WithOpName("Dense_w"), input, m_vars["W"+idx]), m_vars["B"+idx]);
    if(bActivation)
        return Relu(scope.WithOpName("Relu"), dense);
    else
        return dense;
}

Status NeuralNet::FreezeSave(string& file_name)
{
    vector<Tensor> out_tensors;
    //Extract: current weights and biases current values
    TF_CHECK_OK(t_session->Run(v_weights_biases , &out_tensors));
    unordered_map<string, Tensor> variable_to_value_map;
    int idx = 0;
    for(Output o: v_weights_biases)
    {
        variable_to_value_map[o.node()->name()] = out_tensors[idx];
        idx++;
    }
    GraphDef graph_def;
    TF_CHECK_OK(t_root.ToGraphDef(&graph_def));
    //call the utility function (modified)
    SavedModelBundle saved_model_bundle;
    SignatureDef signature_def;
    (*signature_def.mutable_inputs())[input_batch_var.name()].set_name(input_batch_var.name());
    (*signature_def.mutable_outputs())[out_classification.name()].set_name(out_classification.name());
    MetaGraphDef* meta_graph_def = &saved_model_bundle.meta_graph_def;
    (*meta_graph_def->mutable_signature_def())["signature_def"] = signature_def;
    *meta_graph_def->mutable_graph_def() = graph_def;
    SessionOptions session_options;
    saved_model_bundle.session.reset(NewSession(session_options));//even though we will not use it
    GraphDef frozen_graph_def;
    std::unordered_set<string> inputs;
    std::unordered_set<string> outputs;
    TF_CHECK_OK(FreezeSavedModel(saved_model_bundle, &frozen_graph_def, &inputs, &outputs));

    //write to file
    return WriteBinaryProto(Env::Default(), file_name, frozen_graph_def);
}


auto tf_tensor_to_vector(tensorflow::Tensor tensor, int32_t tensorSize) {
  int32_t* tensor_ptr = tensor.flat<int32_t>().data();
  std::vector<int32_t> v(tensor_ptr, tensor_ptr + tensorSize);
  return v;
}

int main(){	
	using namespace tensorflow;
  	using namespace tensorflow::ops;
	
	auto scope = Scope::NewRootScope();

	auto c1 = Const(scope, 10, /* shape */ {2, 2});

   	 // [1 1] * [41; 1]
  	  auto x = MatMul(scope, {{1, 1}}, {{41}, {1}});

  	  ClientSession session(scope);

  	  std::vector<Tensor> outputs;

  	  auto status = session.Run({x}, &outputs);

  	  TF_CHECK_OK(status);

  	  std::cout << "Underlying Scalar value -> " << outputs[0].flat<int>()<< std::endl;
	return 0;
}

