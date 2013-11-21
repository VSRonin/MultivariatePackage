#ifndef AbstarctCopula_h__
#define AbstarctCopula_h__
#include "AbstractDistribution.h"
namespace Multivariate{
	class AbstarctCopula{
	protected:
		AbstarctDistribution* BaseDist;
		AbstarctCopula():BaseDist(NULL){}
		bool CheckCoordinatesInput(const Eigen::VectorXd& Coordinates) const;
	public:
		bool IsValid() const {return BaseDist->IsValid();}
		void SetRandomSeed(unsigned int NewSeed){return BaseDist->SetRandomSeed(NewSeed);}
		unsigned int GetCurrentSeed()const{return BaseDist->GetCurrentSeed();}
		unsigned int GetDimension() const {return BaseDist->GetDimension();}
		//! Set the dimensionality of the distribution
		/*!
		\param Dimension the new dimensionality of the distribution 
		\return A boolean determining if the dimensionality was changed successfully
		\details This function will try to change the dimensionality of the distribution (e.g. 2 for bivariate, 3 for trivariate, etc.)
	
		The variance covariance matrix will default to an identity matrix

		If the argument passed is less than 2 the dimensionality will not be changed and the function will return false

		\sa GetDimension()
		*/
		bool SetDimension(unsigned int Dimension){if(Dimension>1U) return BaseDist->SetDimension(Dimension); else return false;}
		//! Generates multiple simulations from the copula
		/*!
		\param NumSamples The number of simulation to run
		\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
		\details This function generates NumSamples simulation from the current copula and returns them in matrix form.
	
		If NumSamples is 0 or the distribution is invalid, a null matrix is returned
		*/
		virtual Eigen::MatrixXd ExtractSamples(unsigned int NumSamples)const =0;
		virtual double GetDensity(const Eigen::VectorXd& Coordinates)const =0;
		virtual double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const =0;
		virtual Eigen::VectorXd GetQuantile(double Prob)const =0;
		Eigen::RowVectorXd ExtractSample() const{if(BaseDist->GetDimension()>1U) return ExtractSamples(1U); else return Eigen::RowVectorXd();}
		std::vector<double> ExtractSampleVector() const;
		std::map<unsigned int,std::vector<double> > ExtractSamplesMap(unsigned int NumSamples) const;
		double GetDensity(const std::vector<double>& Coordinates)const;
		double GetCumulativeDesity(const std::vector<double>& Coordinates)const;
		std::vector<double> GetQuantileVector(double Prob)const;
#ifdef mvNormSamplerUnsafeMethods
		double* GetQuantileArray(double Prob);
		double GetCumulativeDesity(double* Coordinates)const;
		double GetDensity(double* Coordinates)const;
		double* ExtractSampleArray() const;
		double** ExtractSamplesMatix(unsigned int NumSamples) const;
#endif
	};
}
#endif // AbstarctCopula_h__