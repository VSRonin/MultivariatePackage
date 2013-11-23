#include "ClaytonCopula.h"
#include <boost/math/distributions/gamma.hpp>
#include <boost/random/uniform_real_distribution.hpp>
using namespace Multivariate;
ClaytonCopula::ClaytonCopula(unsigned int Dimension)
	:AbstractArchimedean(Dimension,1.0)
{
	CheckValidity();
}
ClaytonCopula::ClaytonCopula(unsigned int Dimension,double theta)
	:AbstractArchimedean(Dimension,theta)
{
	CheckValidity();
}
double ClaytonCopula::GeneratorFunction(double x)const{
	return pow(x,-Theta)-1.0;
}
double ClaytonCopula::GeneratorInverseFunction(double x)const{
	return pow(1.0+x,-(1.0/Theta));
}
double ClaytonCopula::GeneratorFunctionDerivative(double x)const{
	return -Theta*pow(x,-Theta-1.0);
}
double ClaytonCopula::GeneratorInverseFunctionDerivative(double x)const{
	return -pow(x+1.0,-(Theta+1)/Theta)/Theta;
}
bool ClaytonCopula::CheckValidity(){
	if(Dim>1U && Theta>0.0) AllValid=true;
	else AllValid=false;
	return AllValid;
}
Eigen::VectorXd ClaytonCopula::GetQuantile(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return Eigen::VectorXd();
	if(Prob==1.0 || Prob==0.0){
		Eigen::VectorXd TempVector(Dim);
		for(unsigned int i=0;i<Dim;i++){
			TempVector(i)= Prob>0.0 ? 1.0 : 0.0;
		}
		return TempVector;
	}
	ClaytonCopula CentralDistr(Dim,Theta);
	CentralDistr.SetRandomSeed(CurrentSeed);
	CentralDistr.ProbToFind=Prob;
	double CenteredQuantile =  boost::math::tools::newton_raphson_iterate(CentralDistr,0.0,0.0,1.0,8);
	Eigen::VectorXd CoordinatesVector=Eigen::VectorXd::Ones(Dim);
	return CenteredQuantile*CoordinatesVector;
}

double ClaytonCopula::SimulateGeneratorInverseFourier()const{
	boost::math::gamma_distribution<double> GammaDist(1.0/Theta,1.0);
	boost::random::uniform_real_distribution<double> dist(0.0,1.0);
	return quantile(GammaDist,dist(RandNumGen));
}