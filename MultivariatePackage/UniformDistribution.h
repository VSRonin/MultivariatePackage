#ifndef UniformDistribution_h__
#define UniformDistribution_h__
#include "AbstractDistribution.h"
namespace Multivariate{
	class UniformDistribution : public AbstarctDistribution{
	private:
		Eigen::MatrixX2d Limits;
	public:
		UniformDistribution(unsigned int Dimension=1U);
		UniformDistribution(unsigned int Dimension, const Eigen::MatrixX2d& MinMax);
		bool SetDimension(unsigned int Dimension);
		const Eigen::MatrixX2d& GetLimits() const {return Limits;}
		virtual bool SetLimits(const Eigen::MatrixX2d& MinMax);
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
		double GetDensity()const;
		double GetDensity(const Eigen::VectorXd& Coordinates)const {return GetDensity();}
		virtual double GetDensity(const std::vector<double>& Coordinates)const {return GetDensity();}
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const;
		Eigen::VectorXd GetQuantile(double Prob)const;
		Eigen::MatrixXd ExtractSamplesCDF(unsigned int NumSamples) const;

		using Multivariate::AbstarctDistribution::ExtractSample;
		using Multivariate::AbstarctDistribution::ExtractSampleVector;
		using Multivariate::AbstarctDistribution::ExtractSamplesMap;
		using Multivariate::AbstarctDistribution::GetDensity;
		using Multivariate::AbstarctDistribution::GetCumulativeDesity;
		using Multivariate::AbstarctDistribution::GetQuantileVector;
		using Multivariate::AbstarctDistribution::ExtractSampleCDF;
		using Multivariate::AbstarctDistribution::ExtractSampleCDFVect;
		using Multivariate::AbstarctDistribution::ExtractSamplesCDFMap;
#ifdef mvNormSamplerUnsafeMethods
		using Multivariate::AbstarctDistribution::GetQuantileArray;
		using Multivariate::AbstarctDistribution::ExtractSampleArray;
		using Multivariate::AbstarctDistribution::ExtractSamplesMatix;
		using Multivariate::AbstarctDistribution::ExtractSampleCDFArray;
		using Multivariate::AbstarctDistribution::ExtractSamplesCDFMatix;
#endif
	};
	class IndependenceCopula : public UniformDistribution{
		private:
			IndependenceCopula(unsigned int Dimension, const Eigen::MatrixX2d& MinMax) : UniformDistribution(Dimension,MinMax){}
			bool SetLimits(const Eigen::MatrixX2d& MinMax){return UniformDistribution::SetLimits(MinMax);}
		public:
			IndependenceCopula(unsigned int Dimension=1U) : UniformDistribution(Dimension){}
	};
}
#endif // UniformDistribution_h__
