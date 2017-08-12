#include <iostream>
#include <armadillo>
#include <vector>
#include <utility>
#include <tuple>
#include <memory>
#include <sstream>
#include <list>
#include <random>
#include <cmath>
using namespace std;
using namespace arma;

//Default epsilon value 
double epsilon = 0.01;


class Network{
public:
  /*
  Constructors
  */

  //Initialize the input using size provided
  Network(const tuple<int,int>&,const tuple<int,int>&,const string&,const double&,const double& ); 

  //Network(Mat& input); Implement this constructor as well

  Network(const Network&); //Copy Constructor



  auto addFCLayer(const tuple<int,int>&); //add a layer of specified size and respective weights initialized from uniform random distribution
  auto add3DConvLayer(const tuple<int,int,int>&); //add a convolutional layer

  auto addFCLayer(Network&); //Add pre-Exisiting network to this layer
  auto setIO(const Mat<double>& , const Mat<double>&);


  Mat<double>& relu(Mat<double>&); //relu activation function

  auto reluGrad(Mat<double>&); //Gradient of relu 

  auto softmax(Mat<double>&); //relu activation function
  /*
    Overload this function or add more parameters for making it dimension friendly
  */
  auto forwardProp(); //For forward propagating the Network
  auto backProp(); //To backpropagate the network
  auto momentum(const int& ,const double&);

  auto gradientDescent(const int&,const int&); 

  double cost(); //To calculate the cost function
  double accuracy();
  auto displayDimensions(); //To display the dimensions of the Network
  /* Debug function to access private members 
    DELETE THIS WHEN DONE DEBUGGING */
  auto debug();
private:
  shared_ptr<vector<Mat<double>>> layerInputs; // input of each Layer

  shared_ptr<vector<Mat<double>>> layerWeights; //  weights of each layer

  shared_ptr<vector<Mat<double>>> layerActivations; //activations for each Layer

  shared_ptr<vector<Mat<double>>> layerSigmas; // derivatives w.r.t layers(inputs)

  shared_ptr<vector<Mat<double>>> layerGradients; //derivatives w.r.t weights

  Mat<double> outputs;                         //the target values

  double lambda;                               //Regularization Parameter
  double alpha;                                // learning Rate
 // shared_ptr<vector<Mat<double>>> layerBiases; // the bias units for each layer
  string activation; // the activaion function to be used
};