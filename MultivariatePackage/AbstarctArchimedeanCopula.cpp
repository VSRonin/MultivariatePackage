#include "AbstarctArchimedeanCopula.h"
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
using namespace Multivariate;
void AbstractArchimedean::SetRandomSeed(unsigned int NewSeed){
	CurrentSeed=NewSeed;
	RandNumGen.seed(CurrentSeed);
}
bool AbstractArchimedean::CheckCoordinatesInput(const Eigen::VectorXd& Coordinates) const{
	if(Coordinates.rows()!=Dim) return false;
	for(unsigned int i=0;i<Dim;i++){
		if(Coordinates(i)<0.0 || Coordinates(i)>1.0) return false;
	}
	return true;
}
double AbstractArchimedean::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	if(!AllValid || !CheckCoordinatesInput(Coordinates)) return -1.0;
	double RunnungSum=0.0;
	for(unsigned int i=0;i<Dim;i++){
		RunnungSum+=GeneratorFunction(Coordinates(i));
	}
	return GeneratorInverseFunction(RunnungSum);
}
double AbstractArchimedean::GetDensity(const Eigen::VectorXd& Coordinates)const{
	if(!AllValid || !CheckCoordinatesInput(Coordinates)) return -1.0;
	double SumGenerator=0.0;
	double SumGeneratorDeriv=0.0;
	for(unsigned int i=0;i<Dim;i++){
		SumGenerator+=GeneratorFunction(Coordinates(i));
		SumGeneratorDeriv+=GeneratorFunctionDerivative(Coordinates(i));
	}
	return GeneratorInverseFunctionDerivative(SumGenerator)*SumGeneratorDeriv;
}
boost::math::tuple<double, double> AbstractArchimedean::operator()(double x){
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++) CoordinatesVector(i)=x;
	return boost::math::make_tuple(GetCumulativeDesity(CoordinatesVector)-ProbToFind,GetDensity(CoordinatesVector));
}
std::vector<double> AbstractArchimedean::ExtractSampleVector() const{
	std::vector<double> Result(Dim);
	if(!AllValid) return Result;
	Eigen::RowVectorXd TempVector=ExtractSample();
	for(unsigned int i=0;i<Dim;i++){
		Result[i]=TempVector(i);
	}
	return Result;
}
bool AbstractArchimedean::CheckValidity(){
	if(Dim>1U) AllValid=true;
	else AllValid=false;
	return AllValid;
}
std::map<unsigned int,std::vector<double> > AbstractArchimedean::ExtractSamplesMap(unsigned int NumSamples) const{
	if(!AllValid || NumSamples==0) return std::map<unsigned int,std::vector<double> >();
	std::map<unsigned int,std::vector<double> > Result;
	std::vector<double> Series(NumSamples);
	Eigen::MatrixXd TempMatrix=ExtractSamples(NumSamples);
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<NumSamples;j++){
			Series[j]=TempMatrix(j,i);
		}
		Result.insert(std::pair<unsigned int,std::vector<double> >(i,Series));
	}
	return Result;
}
double AbstractArchimedean::GetDensity(const std::vector<double>& Coordinates)const{
	if(Coordinates.size()!=Dim) return 0.0;
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates.at(i);
	}
	return GetDensity(TempVector);
}
double AbstractArchimedean::GetCumulativeDesity(const std::vector<double>& Coordinates)const{
	if(Coordinates.size()!=Dim) return 0.0;
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates.at(i);
	}
	return GetCumulativeDesity(TempVector);
}
std::vector<double> AbstractArchimedean::GetQuantileVector(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return std::vector<double>();
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	std::vector<double> Result(Dim);
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVector(i);
	return Result;
}
AbstractArchimedean::AbstractArchimedean(unsigned int Dimension,double theta)
	:Dim(Dimension)
	,Theta(theta)
{
	CheckValidity();
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
Eigen::MatrixXd AbstractArchimedean::ExtractSamples(unsigned int NumSamples)const{
	if(!AllValid || NumSamples<1U) return Eigen::MatrixXd();
	boost::random::uniform_real_distribution<double> Unif(0.0, 1.0);
	Eigen::MatrixXd Result(NumSamples,Dim);
	for(unsigned int j=0;j<Dim;j++){
		for(unsigned int i=0;i<NumSamples;i++){
			Result(i,j)=GeneratorInverseFunction(-log(Unif(RandNumGen))/SimulateGeneratorInverseFourier());
		}
	}
	return Result;
}
#ifdef mvNormSamplerUnsafeMethods
double* AbstractArchimedean::ExtractSampleArray()const{
	if(!AllValid) return NULL;
	Eigen::RowVectorXd TempVector=ExtractSample();
	double* Result=new double[Dim];
	for(unsigned int i=0;i<Dim;i++){
		Result[i]=TempVector(i);
	}
	return Result;
}
double** AbstractArchimedean::ExtractSamplesMatix(unsigned int NumSamples) const{
	if(NumSamples==0 || !AllValid) return NULL;
	Eigen::MatrixXd TempMatrix=ExtractSamples(NumSamples);
	double** Result=new double*[NumSamples];
	for(unsigned int i=0;i<NumSamples;i++){
		Result[i]=new double[Dim];
		for(unsigned int j=0;j<Dim;j++){
			Result[i][j]=TempMatrix(i,j);
		}
	}
	return Result;
}
double AbstractArchimedean::GetCumulativeDesity(double* Coordinates)const{
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates[i];
	}
	return GetCumulativeDesity(TempVector);
}
double* AbstractArchimedean::GetQuantileArray(double Prob){
	if(!AllValid || abs(Prob)>1.0) return NULL;
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	double* Result=new double[Dim];
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVector(i);
	return Result;
}
double AbstractArchimedean::GetDensity(double* Coordinates)const{
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates[i];
	}
	return GetDensity(TempVector);
}
#endif