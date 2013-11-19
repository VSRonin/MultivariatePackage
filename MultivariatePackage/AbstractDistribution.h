#include <Eigen/Dense>
#include <vector>
#include <map>
#include <boost/random/mersenne_twister.hpp>
namespace Multivariate{
	class AbstarctDistribution{
	protected:
		AbstarctDistribution(unsigned int Dimension=1U);
		mutable boost::random::mt19937 RandNumGen;
		unsigned int Dim;
		bool AllValid;
		unsigned int CurrentSeed;
	public:
		bool IsValid() const {return AllValid;}
		void SetRandomSeed(unsigned int NewSeed);
		unsigned int GetCurrentSeed()const{return CurrentSeed;}
		unsigned int GetDimension() const {return Dim;}


		virtual bool SetDimension(unsigned int Dimension) =0;
		virtual Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const =0;
		virtual double GetDensity(const Eigen::VectorXd& Coordinates)const =0;
		virtual double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const =0;
		virtual Eigen::MatrixXd ExtractSamplesCDF(unsigned int NumSamples) const =0;
		virtual Eigen::VectorXd GetQuantile(double Prob)const =0;

		virtual Eigen::RowVectorXd ExtractSample() const{return ExtractSamples(1U);}
		virtual std::vector<double> ExtractSampleVector() const;
		virtual std::map<unsigned int,std::vector<double> > ExtractSamplesMap(unsigned int NumSamples) const;
		virtual double GetDensity(const std::vector<double>& Coordinates)const;
		virtual double GetCumulativeDesity(const std::vector<double>& Coordinates)const;
		virtual std::vector<double> GetQuantileVector(double Prob)const;
		virtual Eigen::RowVectorXd ExtractSampleCDF() const{return ExtractSamples(1U);}
		virtual std::vector<double> ExtractSampleCDFVect() const;
		virtual std::map<unsigned int,std::vector<double> > ExtractSamplesCDFMap(unsigned int NumSamples) const;

		/*
		Convenience template to copy in derived classes to access overloaded functions

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
		*/

#ifdef mvNormSamplerUnsafeMethods
		virtual double* GetQuantileArray(double Prob);
		virtual double GetCumulativeDesity(double* Coordinates)const;
		virtual double GetDensity(double* Coordinates)const;
		virtual double* ExtractSampleArray() const;
		virtual double** ExtractSamplesMatix(unsigned int NumSamples) const;
		virtual double* ExtractSampleCDFArray() const;
		virtual double** ExtractSamplesCDFMatix(unsigned int NumSamples) const;
#endif
	};
}