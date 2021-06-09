/*
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
Following tutorial: https://mxnet.apache.org/versions/1.8.0/api/cpp/docs/tutorials/basics
Complete program is given on: https://github.com/researchmm/tasn/blob/master/tasn-mxnet/cpp-package/example/mlp_cpu.cpp
Dependency: "utils.h" should be in the working directory
Compile: $ g++ <file_name>.cpp -lmxnet
*/

// libraries
#include <iostream>
#include <chrono>
#include <mxnet-cpp/MxNetCpp.h>
#include "utils.h"

// namespaces
using namespace std;
using namespace mxnet::cpp;

// global constants
const int BATCH_SIZE = 100;
const int IMAGE_SIZE = 28;
const vector<int> layers{128, 64, 10};
const float LEARNING_RATE = 0.1;
const float WEIGHT_DECAY = 1e-2;
const int MAX_EPOCH = 10;

// defining MLP (Multi-Layer Perceptron)
Symbol mlp(const vector<int> &layers) {
	auto x = Symbol::Variable("X");
	auto label = Symbol::Variable("label");
	vector<Symbol> weights(layers.size());
	vector<Symbol> biases(layers.size());
	vector<Symbol> outputs(layers.size());
	for(int i=0; i<layers.size(); i++) {
		weights[i] = Symbol::Variable("w" + to_string(i));
		biases[i] = Symbol::Variable("b" + to_string(i));
		Symbol fc = FullyConnected(
			i == 0? x : outputs[i - 1],
			weights[i],
			biases[i],
			layers[i]
		);
		outputs[i] = i == layers.size() - 1 ? fc : Activation(fc, ActivationActType::kRelu);
	}
	return SoftmaxOutput(outputs.back(), label);
}

// MAIN function
int main() {
	/* Context:
	MXNet can run operations on CPU and different GPUs.
	A Context describes the device type and ID on which
	computation should be carried on.*/
	Context ctx = Context::cpu();
	// creating data iter
	cout<<"Creating data iter to load the MNIST dataset"<<endl;
	/* vector:
	Vectors are same as dynamic arrays with the ability to resize automatically
	when an element is inserted or deleted, with their storage being handled
	automatically by the container. Vector elements are placed in contiguous
	storage so that they can be accessed and traversed using iterators. In
	vectors, data is inserted at the end.*/
	vector<string> data_files = {
		"./../dataset/mnist_data/train-images.idx3-ubyte",
		"./../dataset/mnist_data/train-labels.idx1-ubyte",
		"./../dataset/mnist_data/t10k-images.idx3-ubyte",
		"./../dataset/mnist_data/t10k-labels.idx1-ubyte"
	};
	/* auto:
	*Type Inference refers to automatic deduction of the data type of an
	expression in a programming language. Before C++ 11, each data type
	needs to be explicitly declared at compile time, limiting the values
	of an expression at runtime but after the new version of C++, many
	keywords are included which allows a programmer to leave the type
	deduction to the compiler itself.
	With type inference capabilities, we can spend less time having to
	write out things compiler already knows. As all the types are deduced
	in compiler phase only, the time for compilation increases slightly
	but it does not affect the run time of the program.*/
	auto train_iter = MXDataIter("MNISTIter");
	setDataIter(&train_iter, "Train", data_files, BATCH_SIZE);
	auto val_iter = MXDataIter("MNISTIter");
	setDataIter(&val_iter, "Label", data_files, BATCH_SIZE);
	/* map:
	Maps are associative containers that store elements in a mapped fashion.
	Each element has a key value and a mapped value. No two mapped values
	can have same key values.*/
	map<string, NDArray> args;
	auto net = mlp(layers);
	args["X"] = NDArray(Shape(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE), ctx);
	args["label"] = NDArray(Shape(BATCH_SIZE), ctx);
	net.InferArgsMap(ctx, &args, args);
	auto initializer = Uniform(0.01);
	for(auto& arg : args) {
		initializer(arg.first, &arg.second);
	}
	Optimizer* opt = OptimizerRegistry::Find("sgd");
	opt->SetParam("rescale_grad", 1.0 / BATCH_SIZE)
		->SetParam("lr", LEARNING_RATE)
		->SetParam("wd", WEIGHT_DECAY);
	auto *exec = net.SimpleBind(ctx, args);
	auto arg_names = net.ListArguments();
	// start training
	for(int iter=0; iter<MAX_EPOCH; iter++) {
		auto tic = chrono::system_clock::now();
		int samples = 0;
		train_iter.Reset();
		while(train_iter.Next()) {
			samples += BATCH_SIZE;
			auto data_batch = train_iter.GetDataBatch();
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);
			exec->Forward(true);
			exec->Backward();
			for(size_t i=0; i<arg_names.size(); i++) {
				if(arg_names[i] == "X" || arg_names[i] == "label")
					continue;
				opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
			}
		}
		auto toc = chrono::system_clock::now();	
		Accuracy acc;
		val_iter.Reset();
		while(val_iter.Next()) {
			auto data_batch = val_iter.GetDataBatch();
			data_batch.data.CopyTo(&args["X"]);
			data_batch.label.CopyTo(&args["label"]);
			exec->Forward(false);
			acc.Update(data_batch.label, exec->outputs[0]);
		}
		float duration = chrono::duration_cast<chrono::milliseconds> (toc - tic).count() / 1000.0;
		cout<<"Epoch: "<<iter<<" Accuracy: "<<acc.Get()<<" Time: "<<duration<<" secs"<<endl;
	}
	delete exec;
	MXNotifyShutdown();
	cout<<"I am working fine!!!"<<endl;
	return 0;
}
