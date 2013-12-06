#include "AbstractDistribution.h"
#include <ctime>
using namespace Multivariate;
AbstarctDistribution::AbstarctDistribution(unsigned int Dimension)
	:Dim(Dimension)
{
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
void AbstarctDistribution::SetRandomSeed(unsigned int NewSeed){
	CurrentSeed=NewSeed;
	RandNumGen.seed(CurrentSeed);
}
std::vector<double> AbstarctDistribution::ExtractSampleVector() const{
	std::vector<double> Result(Dim);
	if(!AllValid) return Result;
	Eigen::RowVectorXd TempVector=ExtractSample();
	for(unsigned int i=0;i<Dim;i++){
		Result[i]=TempVector(i);
	}
	return Result;
}
std::map<unsigned int,std::vector<double> > AbstarctDistribution::ExtractSamplesMap(unsigned int NumSamples) const{
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
double AbstarctDistribution::GetDensity(const std::vector<double>& Coordinates)const{
	if(Coordinates.size()!=Dim) return 0.0;
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates.at(i);
	}
	return GetDensity(TempVector);
}
double AbstarctDistribution::GetCumulativeDesity(const std::vector<double>& Coordinates)const{
	if(Coordinates.size()!=Dim) return 0.0;
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates.at(i);
	}
	return GetCumulativeDesity(TempVector);
}
std::vector<double> AbstarctDistribution::GetQuantileVector(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return std::vector<double>();
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	std::vector<double> Result(Dim);
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVector(i);
	return Result;
}
std::map<unsigned int,std::vector<double> > AbstarctDistribution::ExtractSamplesCDFMap(unsigned int NumSamples) const{
	if(!AllValid || NumSamples<1U) return std::map<unsigned int,std::vector<double> >();
	std::map<unsigned int,std::vector<double> > Result;
	std::vector<double> Series(NumSamples);
	Eigen::MatrixXd TempMatrix=ExtractSamplesCDF(NumSamples);
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<NumSamples;j++){
			Series[j]=TempMatrix(j,i);
		}
		Result.insert(std::pair<unsigned int,std::vector<double> >(i,Series));
	}
	return Result;
}
std::vector<double> AbstarctDistribution::ExtractSampleCDFVect() const{
	if(!AllValid) return std::vector<double>();
	Eigen::RowVectorXd TempVect=ExtractSampleCDF();
	std::vector<double> Result(Dim);
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVect(i);
	return Result;
}
#ifdef mvPackageUnsafeMethods 
double* AbstarctDistribution::ExtractSampleCDFArray() const{
	if(!AllValid) return NULL;
	Eigen::RowVectorXd TempVect=ExtractSampleCDF();
	double* Result=new double[Dim];
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVect(i);
	return Result;
}
double** AbstarctDistribution::ExtractSamplesCDFMatix(unsigned int NumSamples) const{
	if(!AllValid || NumSamples<1U) return NULL;
	double** Result=new double*[NumSamples];
	Eigen::MatrixXd TempMatrix=ExtractSamplesCDF(NumSamples);
	for(unsigned int j=0;j<NumSamples;j++){
		Result[j]=new double[Dim];
		for(unsigned int i=0;i<Dim;i++){
			Result[j][i]=TempMatrix(j,i);
		}
	}
	return Result;
}
double* AbstarctDistribution::ExtractSampleArray()const{
	if(!AllValid) return NULL;
	Eigen::RowVectorXd TempVector=ExtractSample();
	double* Result=new double[Dim];
	for(unsigned int i=0;i<Dim;i++){
		Result[i]=TempVector(i);
	}
	return Result;
}
double** AbstarctDistribution::ExtractSamplesMatix(unsigned int NumSamples) const{
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
double AbstarctDistribution::GetCumulativeDesity(double* Coordinates)const{
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates[i];
	}
	return GetCumulativeDesity(TempVector);
}
double* AbstarctDistribution::GetQuantileArray(double Prob){
	if(!AllValid || abs(Prob)>1.0) return NULL;
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	double* Result=new double[Dim];
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVector(i);
	return Result;
}
double AbstarctDistribution::GetDensity(double* Coordinates)const{
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates[i];
	}
	return GetDensity(TempVector);
}
#endif