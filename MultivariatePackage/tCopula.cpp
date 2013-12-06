#include "tCopula.h"
#include <boost/math/distributions/students_t.hpp>
using namespace Multivariate;
tCopula::tCopula(unsigned int Dimension, unsigned int DegFreedom,const Eigen::MatrixXd& ScalMatr){
	NormalDistribution TempNorm(Dimension,0.0*Eigen::VectorXd::Ones(Dimension>0 ? Dimension:1),ScalMatr);
	BaseDist=new tDistribution(
		Dimension
		,DegFreedom
		,0.0*Eigen::VectorXd::Ones(Dimension>0 ? Dimension:1)
		,TempNorm.IsValid() ? TempNorm.GetCorrelationMatrix() : ScalMatr
	);
	LocalVersion=static_cast<tDistribution*>(BaseDist);
}
bool tCopula::SetScaleMatrix(const Eigen::MatrixXd& ScalMatr){
	NormalDistribution TempNorm(ScalMatr.rows(),0.0*Eigen::VectorXd::Ones(ScalMatr.rows()),ScalMatr);
	if(!TempNorm.IsValid()) return false;
	return LocalVersion->SetScaleMatrix(TempNorm.GetCorrelationMatrix());
}
bool tCopula::SetScaleMatrix(const std::vector<double>& mVect, bool RowWise){
	if(mVect.size()!=LocalVersion->GetDimension()*LocalVersion->GetDimension()) return false;
	Eigen::MatrixXd TempMatrix(LocalVersion->GetDimension(),LocalVersion->GetDimension());
	for(unsigned int i=0;i<mVect.size();i++){
		TempMatrix(i/LocalVersion->GetDimension(),i%LocalVersion->GetDimension())=mVect.at(i);
	}
	if(!RowWise) TempMatrix.transposeInPlace();
	return SetScaleMatrix(TempMatrix);
}
double tCopula::GetDensity(const Eigen::VectorXd& Coordinates)const{
	if(GetDimension()>1U && CheckCoordinatesInput(Coordinates)){
		boost::math::students_t Standardt(LocalVersion->GetDegreesOfFreedom());
		Eigen::VectorXd AdjustedCoords(GetDimension());
		for (unsigned int i=0;i<GetDimension();i++){
			if(Coordinates(i)==0 || Coordinates(i)==1) return -1.0;
			AdjustedCoords(i)=quantile(Standardt,Coordinates(i));
		}
		return LocalVersion->GetDensity(AdjustedCoords); 
	}
	return -1.0;
}
double tCopula::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	if(GetDimension()>1U && CheckCoordinatesInput(Coordinates)){
		boost::math::students_t Standardt(LocalVersion->GetDegreesOfFreedom());
		Eigen::VectorXd AdjustedCoords(GetDimension());
		for (unsigned int i=0;i<GetDimension();i++){
			if(Coordinates(i)==0 || Coordinates(i)==1) return -1.0;
			AdjustedCoords(i)=quantile(Standardt,Coordinates(i));
		}
		return LocalVersion->GetCumulativeDesity(AdjustedCoords); 
	}
	return -1.0;
}
Eigen::VectorXd tCopula::GetQuantile(double Prob)const{
	if(GetDimension()<2U) return Eigen::VectorXd();
	boost::math::students_t Standardt(LocalVersion->GetDegreesOfFreedom());
	Eigen::VectorXd TempVector=LocalVersion->GetQuantile(Prob);
	for(int i=0;i<TempVector.rows();i++){
		TempVector(i)=cdf(Standardt,TempVector(i));
	}
	return TempVector;
}