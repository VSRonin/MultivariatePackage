#include "AbsaractCopula.h"
using namespace  Multivariate;
bool AbstarctCopula::CheckCoordinatesInput(const Eigen::VectorXd& Coordinates) const{
	if(Coordinates.rows()!=BaseDist->GetDimension()) return false;
	for(unsigned int i=0;i<BaseDist->GetDimension();i++){
		if(Coordinates(i)<0.0 || Coordinates(i)>1.0) return false;
	}
	return true;
}
double AbstarctCopula::GetDensity(const std::vector<double>& Coordinates)const{
	if(BaseDist->GetDimension()>1U && BaseDist->IsValid()){
		Eigen::VectorXd TempVec(Coordinates.size());
		for(unsigned int i=0;i<Coordinates.size();i++) TempVec(i)=Coordinates[i];
		return GetDensity(TempVec);
	}
	return -1.0;
}
double AbstarctCopula::GetCumulativeDesity(const std::vector<double>& Coordinates)const{
	if(BaseDist->GetDimension()>1U && BaseDist->IsValid()){
		Eigen::VectorXd TempVec(Coordinates.size());
		for(unsigned int i=0;i<Coordinates.size();i++) TempVec(i)=Coordinates[i];
		return GetCumulativeDesity(TempVec);
	}
	return -1.0;
}
std::vector<double> AbstarctCopula::ExtractSampleVector() const{
	if(!BaseDist->IsValid()) return std::vector<double>();
	Eigen::RowVectorXd TempVec=ExtractSample();
	std::vector<double> Result(BaseDist->GetDimension());
	for(unsigned int i=0;i<BaseDist->GetDimension();i++){
		Result[i]=TempVec(i);
	}
	return Result;
}
std::map<unsigned int,std::vector<double> > AbstarctCopula::ExtractSamplesMap(unsigned int NumSamples) const{
	if(!BaseDist->IsValid() || NumSamples<1U) return std::map<unsigned int,std::vector<double> >();
	std::map<unsigned int,std::vector<double> > Result;
	std::vector<double> TempVector;
	Eigen::MatrixXd Tempmatrix=ExtractSamples(NumSamples);
	for(unsigned int i=0;i<BaseDist->GetDimension();i++){
		TempVector.clear();
		for(unsigned int j=0;j<NumSamples;j++){
			TempVector.push_back(Tempmatrix(j,i));
		}
		Result.insert(std::pair<int,std::vector<double> >(i,TempVector));
	}
	return Result;
}
std::vector<double>  AbstarctCopula::GetQuantileVector(double Prob)const{
	if(!BaseDist->IsValid() || Prob>1.0 || Prob<0.0) return std::vector<double>();
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	std::vector<double> Result(BaseDist->GetDimension());
	for(unsigned int i=0;i<BaseDist->GetDimension();i++) Result[i]=TempVector(i);
	return Result;
}
#ifdef mvNormSamplerUnsafeMethods
double AbstarctCopula::GetCumulativeDesity(double* Coordinates)const{
	if(BaseDist->GetDimension()>1U){
		Eigen::VectorXd TempVec(BaseDist->GetDimension());
		for(unsigned int i=0;i<BaseDist->GetDimension();i++) TempVec(i)=Coordinates[i];
		return GetCumulativeDesity(TempVec);
	}
	return -1.0;
}
double AbstarctCopula::GetDensity(double* Coordinates)const{
	if(BaseDist->GetDimension()>1U){
		Eigen::VectorXd TempVec(BaseDist->GetDimension());
		for(unsigned int i=0;i<BaseDist->GetDimension();i++) TempVec(i)=Coordinates[i];
		return GetDensity(TempVec);
	}
	return -1.0;
}
double* AbstarctCopula::GetQuantileArray(double Prob){
	if(!BaseDist->IsValid() || abs(Prob)>1.0) return NULL;
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	double* Result=new double[BaseDist->GetDimension()];
	for(unsigned int i=0;i<BaseDist->GetDimension();i++) Result[i]=TempVector(i);
	return Result;
}
double* AbstarctCopula::ExtractSampleArray() const{
	if(!BaseDist->IsValid()) return NULL;
	Eigen::RowVectorXd TempVector=ExtractSample();
	double* Result=new double[BaseDist->GetDimension()];
	for(unsigned int i=0;i<BaseDist->GetDimension();i++){
		Result[i]=TempVector(i);
	}
	return Result;
}
double** AbstarctCopula::ExtractSamplesMatix(unsigned int NumSamples) const{
	if(NumSamples==0 || !BaseDist->IsValid()) return NULL;
	Eigen::MatrixXd TempMatrix=ExtractSamples(NumSamples);
	double** Result=new double*[NumSamples];
	for(unsigned int i=0;i<NumSamples;i++){
		Result[i]=new double[BaseDist->GetDimension()];
		for(unsigned int j=0;j<BaseDist->GetDimension();j++){
			Result[i][j]=TempMatrix(i,j);
		}
	}
	return Result;
}
#endif