#include "network.cpp"



int main(){
mat input,output;
input.load("train.csv");
output.load("trainTarget.csv");
int sizei = arma::size(input)(0),siz=arma::size(input)(1),sizeo = arma::size(output)(0),siz1=arma::size(output)(1);
Network net({sizei,siz},{sizeo,siz1},"relu",0.04,0.005);
net.setIO(input,output);

net.addFCLayer({sizei,90});
net.addFCLayer({sizei,6});
net.displayDimensions();
net.momentum(200);
cout << "Accuracy => " << net.accuracy() << endl;
}