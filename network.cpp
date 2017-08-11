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
  
  auto addFCLayer(Network&); //Add pre-Exisiting network to this layer
  auto setIO(const Mat<double>& , const Mat<double>&);


  Mat<double>& relu(Mat<double>&); //relu activation function

  auto reluGrad(Mat<double>&); //Gradient of relu 

  auto softmax(Mat<double>&); //relu activation function
  /*
    Overload this function or add more parameters for making it dimension friendly
  */
  auto forwardProp(); //For forward propagating the Network
  auto backProp();
  auto gradientDescent(const int&); 

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
//Intialize using 3d Mat to retain flexibility for conv nets
Network::Network(const tuple<int,int>& insize,const tuple<int,int>& outsize,const string& act = "relu",const double& lamb = 0.5,const double& alph = 0.1):

layerInputs(make_shared<vector<Mat<double>>>(vector<Mat<double>>({Mat<double>(get<0>(insize),get<1>(insize))}))),
layerWeights(make_shared<vector<Mat<double>>>(vector<Mat<double>>())),
layerActivations(make_shared<vector<Mat<double>>>(vector<Mat<double>>())),
outputs(Mat<double>({Mat<double>(get<0>(outsize),get<1>(outsize))})),
layerSigmas(make_shared<vector<Mat<double>>>(vector<Mat<double>>())),
layerGradients(make_shared<vector<Mat<double>>>(vector<Mat<double>>())),
//layerBiases(make_shared<vector<Mat<double>>>(vector<Mat<double>>())),
activation(act) , lambda(lamb) , alpha(alph)

{
  //Initialization by Assignment
  /*
  layerInputs = make_shared<vector<Mat<double>>>(vector<Mat<double>>({Mat<double>(get<0>(size),get<1>(size),get<2>(size),fill::zeros)})); 
  layerWeights = make_shared<vector<Mat<double>>>(vector<Mat<double>>()); 
  layerActivations = make_shared<vector<Mat<double>>>(vector<Mat<double>>({Mat<double>(get<0>(size),get<1>(size),get<2>(size),fill::zeros)})); 
  layerBiases = make_shared<vector<Mat<double>>>(vector<Mat<double>>()); 
  actiavtion = act;
  */
}

//Constructor.. For details refer the class declaration
Network::Network(const Network& copy){

  layerInputs = copy.layerInputs;
  layerWeights = copy.layerWeights;
  layerActivations = copy.layerActivations;
  //layerBiases = copy.layerBiases;
  activation = copy.activation;
}
//FC layers have 2D. Depth will be set to one
auto Network::addFCLayer(const tuple<int,int>& size){
    //Calclulate the dimensions of weight matrix to propagate to this layer
   auto width = get<1>(size);
   auto height = arma::size(layerInputs->at(layerInputs->size()-1))(1);
    
    // Add Layer and weights

   layerInputs->push_back(Mat<double>(get<0>(size),get<1>(size)));
   Mat<double>weight(height+1,width); 
   //Initialize weights using formula => 4* sqrt(6/(height+1,width))
   auto dist_param = (double)4* sqrt((double)6/(height+1,width));
    std::default_random_engine generator;

   uniform_real_distribution<>distribution(-dist_param,dist_param);
  
   for(auto i = weight.begin();i!=weight.end();i++){
      *i= distribution(generator);
     
   }

  layerWeights->push_back(weight);

}

//Print the dimensions of Entire Network in 3D
auto Network::displayDimensions(){

  cout << "Input Layers :-" << endl;
  for(auto i= layerInputs->begin();i!=layerInputs->end();i++)
      cout << arma::size(*i) << "  ";  

  cout << "\nActivation Layers :-" << endl;
  for(auto i= layerActivations->begin();i!=layerActivations->end();i++)
      cout << arma::size(*i) << "  ";   
  cout << "\n" ;

  cout << "weights :-\n"; 
  for(auto i= layerWeights->begin();i!=layerWeights->end();i++)
      cout << arma::size(*i) << "  ";   
  cout << "\n" ;
  
   cout << "Sigmas :-\n"; 
  for(auto i= layerSigmas->begin();i!=layerSigmas->end();i++)
      cout << arma::size(*i) << "  ";   
  cout << "\n" ;

   cout << "Gradient :-\n"; 
  for(auto i= layerGradients->begin();i!=layerGradients->end();i++)
      cout << arma::size(*i) << "  ";   
  cout << "\n" ;

}

//Relu activation function
Mat<double>& Network::relu(Mat<double>& layer)
{
   layer.for_each([](mat::elem_type& x){x = max(x,epsilon*x);});
   return layer;
}

auto Network::reluGrad(Mat<double>& layer){

  layer.for_each([](mat::elem_type& x){x = (x>0)?1:epsilon;});
  return layer;
}

//SoftMax Activation
auto Network::softmax(Mat<double>& layer){

  double eps = 1e-8;
  Mat<double> maxRows = arma::max(layer,1);
  for(auto i=0; i<layer.n_rows;i++){
    layer.row(i) = layer.row(i)-maxRows(i);
  }
  layer = arma::exp(layer );

  Mat<double>matSum = arma::sum(layer,1);

  for(int i=0;i<layer.n_rows;i++)
  layer.row(i) = layer.row(i)/matSum(i);

/*
double eps = 1e-8;
  layer = arma::exp(layer - layer.max();
  Mat<double>matSum = arma::sum(layer,1);
  cout << "check " << layer(1,0) << " / " << matSum(1) << " = " << layer(1,0)/matSum(1) << endl;   
  for(int i=0;i<layer.n_rows;i++)
  layer.row(i) = layer.row(i)/(matSum(i)+eps);
*/
}


//Forward Prop
auto Network::forwardProp()
{
   // The 1st activation layer is just the input layer with extra bais column
   layerActivations->push_back(layerInputs->at(0));
   layerActivations->at(0).insert_cols(0,Col<double>(arma::size(layerActivations->at(0))[0],fill::ones));


  
   for(auto i=1;i<layerInputs->size();i++){

     layerInputs->at(i) = layerActivations->at(i-1)*layerWeights->at(i-1);
     layerActivations->push_back(layerInputs->at(i));


    //Using Relu actuvation by default if not the output layer otherwise use softmax
    if(i!=layerInputs->size()-1)
      relu(layerActivations->at(i));
    else
      softmax(layerActivations->at(i));


    //Insert a column of bais unit in the activation because it is the input of next layer, If there are more layers
    if(i<layerInputs->size()-1)
      layerActivations->at(i).insert_cols(0,Col<double>(arma::size(layerActivations->at(i))(0),fill::ones));
      
   }


   
   
}

//BackProp
auto Network::backProp(){ 
  //Calculate Sigmas => derivatives w.r.t each layer
  // derivative w.r.t output layer = (Activated output Layer) - (target value)
  layerSigmas->push_back( layerActivations->at(layerActivations->size()-1) - outputs );

  for(auto i = layerActivations->size()-2;i>=1;i--){
    // Add a column of 1s in input layer to account for bais weights
    layerInputs->at(i).insert_cols(0,Col<double>(arma::size(layerInputs->at(i))[0],fill::ones));

    layerSigmas->insert(layerSigmas->begin(),(layerSigmas->at(0)*layerWeights->at(i).t()) % reluGrad(layerInputs->at(i)) );
    layerSigmas->at(0).shed_col(0);

    //Remove the added column
    layerInputs->at(i).shed_col(0);
  }

  for(auto i=0;i<layerActivations->size()-1;i++){
    //Replace bias weights with 0s as they are not regularized
    auto tempWeight = layerWeights->at(i);
    tempWeight.shed_row(0);
    tempWeight.insert_rows(0,Row<double>(arma::size(tempWeight)[1], fill::zeros  )  );

    layerGradients->push_back( (layerActivations->at(i).t()*layerSigmas->at(i) + lambda*tempWeight)/(arma::size(outputs)[0])
                              );
  }

}

//Gradient Descenet
auto Network::gradientDescent(const int& epoch){
 for(auto i=1;i<=epoch;i++){
   // Forward pass , print cost and Backward pass

   forwardProp();
   cout << "iteration " << i << "cost => " << cost() << "accuracy => " << accuracy() <<endl;
   backProp();
   
  //update all weights
   for(auto i = 0;i<layerWeights->size();i++){
      layerWeights->at(i) = layerWeights->at(i) - alpha* layerGradients->at(i);
   }
  //clear the vectors
  if(i<epoch){
  layerActivations->clear();
  layerSigmas->clear();
  layerGradients->clear();
  }
 }
 
}

//Calculate softmax cost function with regularization 
double Network::cost(){
  double eps = 1e-8;
  //-(output*(np.log(a3+eps))+(1-output)*(np.log(1-a3+eps))).sum()
  double cost = sum(sum( -(  outputs % log(layerActivations->at(layerActivations->size()-1)+eps) +
                       (1-outputs) % (log(1-layerActivations->at(layerActivations->size()-1)+eps))  )   ));
  //regularize
  double regularizationCost = 0.0;
  Mat<double> temp;
  for(auto i = layerWeights->begin();i!=layerWeights->end();i++) 
     {
        temp = *i;
        temp.shed_row(0);
        regularizationCost += sum(sum(arma::square(temp)));
     }

  cost = (cost + (lambda/2)*regularizationCost)/(arma::size(outputs)(0));
  
  return cost;
}

double Network::accuracy(){
  auto pred = layerActivations->at(layerActivations->size()-1);
  auto count = 0.0;
  Mat<double> maxRow = arma::max(pred,1);
  for(auto i=0;i<pred.n_rows;i++){
     for(auto j=0;j<pred.n_cols;j++){
        if(pred(i,j)==maxRow(i)&& outputs(i,j)==1.0)
          count++;
        
     }
  }
return (count)/(arma::size(outputs)(0))*100;

}


//Debug function
auto Network::debug(){
  cout << "\nweights" << endl;
  for(auto i =0;i<=5;i++)
   cout << layerWeights->at(0).row(i) << endl;
}

auto Network::setIO(const Mat<double>&a , const Mat<double>&b){
  layerInputs->at(0) = a;
  outputs = b;
}





int main(){
epsilon = 0.01;
arma:mat dataset,out;
dataset.load("train.csv");
out.load("trainTarget.csv");

 vector<pair<double,double>>acc;
 vector<double>costs;
 double lambda = 0.0003,alpha = 0.024 , accuracy = 0,cost = 1000,add = 0.0002;
 //while(true){
    Network net({7352,561},{7352,6},"relu",lambda,alpha);
    net.setIO(dataset,out);
    net.addFCLayer({7352,90});
    net.addFCLayer({7352,6});
    //net.displayDimensions();
    net.gradientDescent(200);
  /*  auto a1 = net.accuracy(), c1 = net.cost();
    cout << "accuracy => " << a1 << endl;
    if(a1>accuracy){
      accuracy = a1;
      cost = c1;
      acc.push_back(make_pair(alpha,accuracy));
      alpha += add;
    }else{
       alpha = acc.at(acc.size()-1).first;
       add /= 10;
       alpha += add;
    }*/
 //}
    return 0;
}