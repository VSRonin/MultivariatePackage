#include "GaussianCopula.h"
#include <boost/math/distributions/normal.hpp>
using namespace Multivariate;
bool GaussianCopula::SetVarCovMatrix(const Eigen::MatrixXd& CovMatr){
	if(CovMatr.rows()!=CovMatr.cols() || CovMatr.rows()!=LocalVersion->GetDimension()) //The Var-Cov Matrix must be squared and have as many rows as there are dimensions
		return false;
	if(CovMatr!=CovMatr.transpose()) //The Var-Cov Matrix must be symmetric
		return false;
	//The Variance Covariance matrix must be positive definite
	if(CovMatr.determinant()<0.0)
		return false;

	Eigen::VectorXcd RelatedEigen=CovMatr.eigenvalues();
	for (unsigned int i=0;i<LocalVersion->GetDimension();i++){
		if(RelatedEigen(i).real()<0.0){ 
			return false;
		}
	}
	Eigen::MatrixXd AdjustdMatrix(LocalVersion->GetDimension(),LocalVersion->GetDimension());
	for(unsigned int i=0;i<LocalVersion->GetDimension();i++){
		for(unsigned int j=0;j<LocalVersion->GetDimension();j++){
			if(i==j) AdjustdMatrix(i,j)=1.0;
			else AdjustdMatrix(i,j)=CovMatr(i,j)/sqrt(CovMatr(i,i)*CovMatr(j,j));
		}
	}
	LocalVersion->SetVarCovMatrix(AdjustdMatrix);
	return true;
}
bool GaussianCopula::SetVarCovMatrix(const std::vector<double>& mVect, bool RowWise){
	if(mVect.size()!=LocalVersion->GetDimension()*LocalVersion->GetDimension()) return false;
	Eigen::MatrixXd TempMatrix(LocalVersion->GetDimension(),LocalVersion->GetDimension());
	for(unsigned int i=0;i<mVect.size();i++){
		TempMatrix(i/LocalVersion->GetDimension(),i%LocalVersion->GetDimension())=mVect.at(i);
	}
	if(!RowWise) TempMatrix.transposeInPlace();
	return SetVarCovMatrix(TempMatrix);
}
double GaussianCopula::GetDensity(const Eigen::VectorXd& Coordinates)const{
	if(GetDimension()>1U && CheckCoordinatesInput(Coordinates)){
		boost::math::normal StandardNormal(0.0,1.0);
		Eigen::VectorXd AdjustedCoords(GetDimension());
		for (unsigned int i=0;i<GetDimension();i++){
			if(Coordinates(i)==0 || Coordinates(i)==1) return -1.0;
			AdjustedCoords(i)=quantile(StandardNormal,Coordinates(i));
		}
		return LocalVersion->GetDensity(AdjustedCoords); 
	}
	return -1.0;
}
double GaussianCopula::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	if(GetDimension()>1U && CheckCoordinatesInput(Coordinates)){
		boost::math::normal StandardNormal(0.0,1.0);
		Eigen::VectorXd AdjustedCoords(GetDimension());
		for (unsigned int i=0;i<GetDimension();i++){
			if(Coordinates(i)==0 || Coordinates(i)==1) return -1.0;
			AdjustedCoords(i)=quantile(StandardNormal,Coordinates(i));
		}
		return LocalVersion->GetCumulativeDesity(AdjustedCoords); 
	}
	return -1.0;
}
Eigen::VectorXd GaussianCopula::GetQuantile(double Prob)const{
	if(GetDimension()<2U) return Eigen::VectorXd();
	boost::math::normal StandardNormal(0.0,1.0);
	Eigen::VectorXd TempVector=LocalVersion->GetQuantile(Prob);
	for(int i=0;i<TempVector.rows();i++){
		TempVector(i)=cdf(StandardNormal,TempVector(i));
	}
	return TempVector;
}