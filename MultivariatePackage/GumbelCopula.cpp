#include "GumbelCopula.h"
#include <boost/math/constants/constants.hpp>
#include "SpecialFunctions.h"
using namespace Multivariate;
GumbelCopula::GumbelCopula(unsigned int Dimension)
	:AbstractArchimedean(Dimension,1.0)
{
	CheckValidity();
}
GumbelCopula::GumbelCopula(unsigned int Dimension,double theta)
	:AbstractArchimedean(Dimension,theta)
{
	CheckValidity();
}
double GumbelCopula::GeneratorFunction(double x)const{
	return pow(-log(x),Theta);
}
double GumbelCopula::GeneratorInverseFunction(double x)const{
	return exp(-pow(x,1.0/Theta));
}
double GumbelCopula::GeneratorFunctionDerivative(double x)const{
	return (Theta*GeneratorFunction(x))/(x*log(x));
}
double GumbelCopula::GeneratorInverseFunctionDerivative(double x)const{
	return GeneratorInverseFunction(x)*pow(x,(1.0/Theta)-1.0)/Theta;
}
bool GumbelCopula::CheckValidity(){
	if(Dim>1U && Theta>=1.0) AllValid=true;
	else AllValid=false;
	return AllValid;
}
Eigen::VectorXd GumbelCopula::GetQuantile(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return Eigen::VectorXd();
	if(Prob==1.0 || Prob==0.0){
		Eigen::VectorXd TempVector(Dim);
		for(unsigned int i=0;i<Dim;i++){
			TempVector(i)= Prob>0.0 ? 1.0 : 0.0;
		}
		return TempVector;
	}
	GumbelCopula CentralDistr(Dim,Theta);
	CentralDistr.SetRandomSeed(CurrentSeed);
	CentralDistr.ProbToFind=Prob;
	double CenteredQuantile =  boost::math::tools::newton_raphson_iterate(CentralDistr,0.0,0.0,1.0,8);
	Eigen::VectorXd CoordinatesVector=Eigen::VectorXd::Ones(Dim);
	return CenteredQuantile*CoordinatesVector;
}

double GumbelCopula::SimulateGeneratorInverseFourier()const{
	return SimulateStable(1.0/Theta,1.0,pow(cos(boost::math::constants::pi<double>()/(Theta*2.0)),Theta),0,RandNumGen());
}