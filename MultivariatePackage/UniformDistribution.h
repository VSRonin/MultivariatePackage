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
		virtual bool SetDimension(unsigned int Dimension);
		const Eigen::MatrixX2d& GetLimits() const {return Limits;}
		virtual bool SetLimits(const Eigen::MatrixX2d& MinMax);
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
		double GetDensity()const;
		double GetDensity(const Eigen::VectorXd& Coordinates)const {return GetDensity();}
		virtual double GetDensity(const std::vector<double>& Coordinates)const {return GetDensity();}
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const;
		Eigen::VectorXd GetQuantile(double Prob)const;
		Eigen::MatrixXd ExtractSamplesCDF(unsigned int NumSamples) const;
		using Multivariate::AbstarctDistribution::GetDensity;
		using Multivariate::AbstarctDistribution::GetCumulativeDesity;
	};
}
#endif // UniformDistribution_h__
