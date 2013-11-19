#ifndef NormalDist_h__
#define NormalDist_h__
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/roots.hpp>
//! Namespace where all the classes of the library are defined
namespace Multivariate{
//! Multivariate Normal Distribution
/*!
\details This class provides the functionality of calculating the probability density value, cumulative probability density value, inverse cumulative probability density and generate random samples from a multivariate normal.

Defining:
	- \f$ k \f$ as the dimensionality of the distribution
	- \f$ \boldsymbol{\mu}=[\mu_1 \cdots \mu_k] \f$ as the mean vector
	- \f$ \boldsymbol{\Sigma}=\begin{bmatrix}
	\sigma^2_1 & \cdots & \sigma_{1,k}\\
	\vdots  & \ddots & \vdots  \\
	\sigma_{k,1} & \cdots & \sigma^2_k
	\end{bmatrix} \f$ as the variance-covariance matrix

The multivariate normal distribution funtion is defined as: \f$ f(\textbf{x})=((2\pi)^{-\frac{k}{2}} |\boldsymbol{\Sigma}|^{-\frac{1}{2}} e^{(-\frac{1}{2} (\textbf{x}-\boldsymbol{\mu})' \boldsymbol{\Sigma}^{-1} (\textbf{x}-\boldsymbol{\mu}))} \f$

The analytical process for computing pdf and simulate from the distribution is based upon the [mvtnorm package](http://cran.r-project.org/web/packages/mvtnorm/index.html) for [R](http://www.r-project.org/) by Alan Genz, Frank Bretz, Tetsuhisa Miwa, Xuefei Mi, Friedrich Leisch, Fabian Scheipl, Bjoern Bornkamp, Torsten Hothorn
The algorithm for cdf calculation is based on [A. Genz (1992)](http://www.math.wsu.edu/faculty/genz/homepage)

To generate samples a [boost::random::mt19937](http://www.boost.org/doc/libs/1_55_0/doc/html/boost/random/mt19937.html) random number generator is used and seeded with [std::time(NULL)](http://www.cplusplus.com/reference/ctime/time/).<br>
If you construct multiple instances of this class, to avoid the generated samples to be the same, you should supply a different seed. To do so, for example, you can call `MyDistribution.SetRandomSeed(MyDistribution.GetCurrentSeed()+1U);`

Please refer to the \ref examples page for usage examples.

\remark This class is re-entrant
\date November 2013
\license This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.<br><br>
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.<br><br>
You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see [gnu.org](http://www.gnu.org/licenses/).
*/
class NormalDistribution{
private:
	mutable boost::random::mt19937 RandNumGen;
	unsigned int Dim;
	Eigen::VectorXd meanVect;
	Eigen::MatrixXd VarCovMatrix;
	bool AllValid;
	bool CheckValidity();
	NormalDistribution(const NormalDistribution& a);
	NormalDistribution& operator=(const NormalDistribution& a);
	unsigned int CurrentSeed;
	std::vector<unsigned int> FillOrder(const Eigen::VectorXd& source)const;
	double ProbToFind;
	boost::math::tuple<double, double> operator()(double x);
public:
	//! Construct a multivariate normal with the given parameters
	/*!
	\param Dimension The dimensionality of the multivariate normal (supports also univariate gaussian distributions in case this is 1)
	\param mVect The column vector of expected values
	\param CovMatr The variance-covariance
	\details Construct a multivariate normal distribution.
	
	In case:
	- The Dimension is 0
	- The mean vector has a number of elements different from the Dimension
	- The variance-covariance is not square
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension
	
	The class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.
	*/
	NormalDistribution(unsigned int Dimension,const Eigen::VectorXd& mVect,const Eigen::MatrixXd& CovMatr);
	//! Construct a multivariate standard normal
	/*!
	\param Dimension The dimensionality of the multivariate normal (supports also univariate Gaussian distributions in case this is 1)
	\details Construct a multivariate normal distribution with all mean values set to 0, unitary variances and null covariates.
	
	In case The Dimension is 0 the class will be considered invalid (it can be checked using `IsValid()`) and won't produce any result until the problem is fixed.

	If dimension is unspecified, a univariate standard normal is constructed
	*/
	NormalDistribution(unsigned int Dimension=1U);
	//! Check if the distribution is valid
	/*!
	\return A boolean determining if the structure of the distribution is valid
	\details If this function returns false the structure of the distribution is meaningless and no result will be produced until the invalid parameters are cured.
	*/
	bool IsValid() const {return AllValid;}
	//! Set the random number generator seed
	/*!
	\param NewSeed the new random seed
	\note This seed is different from the srand() seed and, even with the same seed, random number generated by the internal generator will be different by those generated by rand()
	\sa GetCurrentSeed()
	*/
	void SetRandomSeed(unsigned int NewSeed);
	//! Get the random number generator seed
	/*!
	\return The random number generator seed.
	\details This function return the seed that was used to initialize the random number generator.
	\note This seed is different from the srand() seed and, even with the same seed, random number generated by the internal generator will be different by those generated by rand()
	\sa SetRandomSeed()
	*/
	unsigned int GetCurrentSeed()const{return CurrentSeed;}
	//! Set the expected values vector
	/*!
	\param mVect the vector of new values for the mean vector
	\return A boolean determining if the mean vector was changed successfully
	\details This function attempts to set the expected values vector to the new values supplied.
	
	If the dimension of the vector is different from the dimension of the distribution the mean vector is not changed and false is returned
	\sa GetMeanVector()
	*/
	bool SetMeanVector(const Eigen::VectorXd& mVect);
	//! Set the expected values vector
	/*!
	\param mVect the vector of new values for the location vector
	\details This is an overloaded version of SetMeanVector(const Eigen::VectorXd&)
	*/
	bool SetMeanVector(const std::vector<double>& mVect);
	//! Set the dimensionality of the distribution
	/*!
	\param Dimension the new dimensionality of the distribution 
	\return A boolean determining if the dimensionality was changed successfully
	\details This function will try to change the dimensionality of the distribution (e.g. 2 for bivariate, 3 for trivariate, etc.)
	
	All the components of the mean vector will be reset to 0 and the variance covariance matrix will default to an identity matrix

	If the argument passed is 0 the dimensionality will not be changed and the function will return false

	\sa GetDimension()
	*/
	bool SetDimension(unsigned int Dimension);
	//! Set the Var-Cov matrix of the distribution
	/*!
	\param CovMatr the new variance covariance matrix of the distribution 
	\return A boolean determining if the variance covariance matrix of the distribution was changed successfully
	\details This function tries to set the variance covariance matrix of the distribution to the new one.

	In case:
	- The variance-covariance is not square
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension

	The variance covariance matrix of the distribution will not be changed and this function will return false

	\sa GetVarCovMatrix()
	*/
	bool SetVarCovMatrix(const Eigen::MatrixXd& CovMatr);
	//! Set the Var-Cov matrix of the distribution
	/*!
	\param mVect a vector containing the elements of the new variance covariance matrix of the distribution
	\param RowWise if it's set to true (the default) the matrix will be filled by row. If it's false it will be filled by columns
	\return A boolean determining if the variance covariance matrix of the distribution was changed successfully
	\details This function tries to set the variance covariance matrix of the distribution to the new one.

	Constructs square a matrix with number of rows equal to the dimensionality of the distribution, it is then filled with the values supplied in order according to the RowWise parameter

	In case:
	- The vector size is different from the square of the distribution dimensionality
	- The variance-covariance is not symmetric
	- The variance-covariance is not semi-positive definite
	- The variance-covariance has a number of rows different from the Dimension

	The variance covariance matrix of the distribution will not be changed and this function will return false

	\sa GetVarCovMatrix()
	*/
	bool SetVarCovMatrix(const std::vector<double>& mVect, bool RowWise=true);
	//! Set the Var-Cov matrix of the distribution
	/*!
	\param CorrelationMatrix The correlation coefficients matrix
	\param Variances the vector of variances
	\return A boolean determining if the variance covariance matrix of the distribution was changed successfully
	\details This function tries to set the variance covariance matrix of the distribution according to the correlation matrix supplied.

	Constructs the variance covariance matrix of the distribution using Variances as diagonal elements and covariates corresponding to the linear correlation coefficient supplied

	In case:
	- The size of the vector of variances is different from the distribution dimensionality
	- The correlation matrix is not symmetric
	- The correlation matrix has elements different from 1 on the diagonal
	- The correlation matrix has a number of rows different from the Dimension
	- The correlation matrix is not squared
	- The correlation matrix has off-diagonal elements greater than 1 in absolute value

	The variance covariance matrix of the distribution will not be changed and this function will return false

	\sa GetVarCovMatrix()
	*/
	bool SetVarCovMatrix(const Eigen::MatrixXd& CorrelationMatrix, const Eigen::VectorXd& Variances);
	//! Get the dimensionality of the distribution
	/*!
	\return The current dimensionality of the distribution
	\sa SetDimension()
	*/
	unsigned int GetDimension() const {return Dim;}
	//! Get the expected values vector of the distribution
	/*!
	\return The current expected values vector of the distribution
	\sa SetMeanVector(const Eigen::VectorXd&)
	*/
	const Eigen::VectorXd& GetMeanVector() const {return meanVect;}
	//! Get the Var-Cov matrix of the distribution
	/*!
	\return The current variance-covariance matrix of the distribution
	\sa SetVarCovMatrix(const Eigen::MatrixXd&)
	\sa SetVarCovMatrix(const std::vector<double>&,bool)
	\sa SetVarCovMatrix(const Eigen::MatrixXd&, const Eigen::VectorXd&)
	*/
	const Eigen::MatrixXd& GetVarCovMatrix() const {return VarCovMatrix;}
	//! Get the linear correlation matrix
	/*!
	\return The linear correlation matrix associated with the current variance-covariance matrix of the distribution
	\sa GetVarCovMatrix()
	*/
	Eigen::MatrixXd GetCorrelationMatrix() const;
	//! Generates a single simulation from the distribution
	/*!
	\return A vector with number of elements equal to the dimensionality of the distribution representing a single extraction from the distribution
	\details This is equal to calling `ExtractSamples(1U)`
	\sa ExtractSamples()
	*/
	Eigen::RowVectorXd ExtractSample() const{return ExtractSamples(1U);}
	//! Generates a single simulation from the distribution
	/*!
	\return A vector with number of elements equal to the dimensionality of the distribution representing a single extraction from the distribution
	\details This is equivalent to ExtractSample() but returns an std::vector intead of an Eigen::RowVectorXd
	\sa ExtractSamples()
	*/
	std::vector<double> ExtractSampleVector() const;
	//! Generates multiple simulations from the distribution
	/*!
	\param NumSamples The number of simulation to run
	\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
	\details This function generates NumSamples simulation from the current distribution and returns them in matrix form.
	
	If NumSamples is 0 or the distribution is invalid, a null matrix is returned
	*/
	Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
	//! Generates multiple simulations from the distribution
	/*!
	\param NumSamples The number of simulation to run
	\return A map that has as keys the index of the dimension (starting from 0) and as values a vector containing the simulation results for that dimension
	\details This function generates NumSamples simulation from the current distribution and returns them in a map form.
	
	If NumSamples is 0 or the distribution is invalid, an empty map is returned
	*/
	std::map<unsigned int,std::vector<double> > ExtractSamplesMap(unsigned int NumSamples) const;
	//! Generates a single simulation from the distribution and returns its marginal CDF
	/*!
	\return A vector with number of elements equal to the dimensionality of the distribution representing a single extraction from the distribution
	\details This is equal to calling `ExtractSamplesCDF(1U)`
	\sa ExtractSamplesCDF()
	*/
	Eigen::RowVectorXd ExtractSampleCDF() const{return ExtractSamples(1U);}
	//! Generates a single simulation from the distribution and returns its marginal CDF
	/*!
	\return A vector with number of elements equal to the dimensionality of the distribution representing a single extraction from the distribution
	\details This is equivalent to ExtractSampleCDF() but returns an std::vector intead of an Eigen::RowVectorXd
	\sa ExtractSamplesCDF()
	*/
	std::vector<double> ExtractSampleCDFVect() const;
	//! Extracts samples from the distribution and returns their marginal CDF
	/*!
	\param NumSamples The number of simulation to run
	\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
	\details This function generates NumSamples simulation from the current distribution, computes the marginal cumulative density function for each of them and returns them in matrix form.
	
	This function simulates extractions from a Gaussian copula

	If NumSamples is 0 or the distribution is invalid, a null matrix is returned
	 */
	Eigen::MatrixXd ExtractSamplesCDF(unsigned int NumSamples) const;
	//! Extracts samples from the distribution and returns their marginal CDF
	/*!
	\param NumSamples The number of simulation to run
	\return A matrix with columns equal to the dimensionality of the distribution and rows equal to the number of simulations
	\details This function generates NumSamples simulation from the current distribution, computes the marginal cumulative density function for each of them and returns them in a map form.
	
	This function simulates extractions from a Gaussian copula

	If NumSamples is 0 or the distribution is invalid, a null matrix is returned
	 */
	std::map<unsigned int,std::vector<double> > ExtractSamplesCDFMap(unsigned int NumSamples) const;
	//! Computes the probability density function of the distribution in correspondence of the supplied coordinates
	/*!
	\param Coordinates A vector containing the coordinates of the point for which the pdf should be computed
	\param GetLogDensity If set to true the log density is returned instead of the actual density
	\return The value of the probability density function
	\details This function computes the probability density function of the current distribution associated with the given coordinates.
	
	If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned
	*/
	double GetDensity(const Eigen::VectorXd& Coordinates, bool GetLogDensity=false)const;
	//! Computes the probability density function of the distribution in correspondence of the supplied coordinates
	/*!
	\param Coordinates  A vector containing the coordinates of the point for which the pdf should be computed
	\param GetLogDensity If set to true the log density is returned instead of the actual density
	\return The value of the probability density function
	\details This is an overloaded version of GetDensity(const Eigen::VectorXd& Coordinates, bool GetLogDensity)const
	*/
	double GetDensity(const std::vector<double>& Coordinates, bool GetLogDensity=false)const;
	//! Computes the cumulative density function of the distribution in correspondence of the supplied coordinates
	/*!
	\param Coordinates A vector containing the coordinates of the point for which the cdf should be computed
	\param UseGenz If set to true the algorithm described in Genz (1992) to calculate the cdf, otherwise it will use full monte-carlo estimation (much slower)
	\param NumSimul The maximum number of simulations for the Genz algorithm. If UseGenz is false this is the number of simulations that will be run by monte-carlo
	\return The value of the cumulative density function
	\details This is an overloaded version of GetCumulativeDesity(const Eigen::VectorXd&, bool, unsigned int)const
	*/
	double GetCumulativeDesity(const std::vector<double>& Coordinates, bool UseGenz=true, unsigned int NumSimul=500000)const;
	//! Computes the cumulative density function of the distribution in correspondence of the supplied coordinates
	/*!
	\param Coordinates A vector containing the coordinates of the point for which the cdf should be computed
	\param UseGenz If set to true the algorithm described in Genz (1992) to calculate the cdf, otherwise it will use full monte-carlo estimation (much slower)
	\param NumSimul The maximum number of simulations for the Genz algorithm. If UseGenz is false this is the number of simulations that will be run by monte-carlo
	\return The value of the cumulative density function
	\details This function computes the cumulative density function of the current distribution associated with the given coordinates.

	If the number of elements in Coordinates is different from the dimensionality of the distribution or the distribution is invalid, -1 is returned.

	If UseGenz is true, the algorithm used is the one described in [Alan Genz - "Numerical Computation of Multivariate Normal Probabilities", Journal of Computational and Graphical Statistics, 1(1992), pp. 141-149](http://www.math.wsu.edu/faculty/genz/papers/mvn.pdf)
	*/
	double GetCumulativeDesity(const Eigen::VectorXd& Coordinates, bool UseGenz=true, unsigned int NumSimul=500000)const;
	//! Computes the inverse cumulative density function of the distribution in correspondence of the supplied probability
	/*!
	\param Prob The probability for which the corresponding quantile must be found
	\return A vector containing the coordinates of the quantile
	\details This function computes the inverse cumulative density function of the current distribution associated with the given probability.
	
	The solution is unique only in the univariate case.<br>
	Generally the system of equations \f$ F^{-1}(Coordinates_1 \cdots Coordinates_Dimensions)=Prob \f$ has Dimensions-1 degrees of freedom.<br>
	The additional restriction imposed to get to an unique solution is that each coordinate has equal distance from it's mean.

	If the probability supplied is greater, in absolute value, than 1 or the distribution is invalid, an empty vector is returned.
	*/
	Eigen::VectorXd GetQuantile(double Prob)const;
	//! Computes the inverse cumulative density function of the distribution in correspondence of the supplied probability
	/*!
	\return A vector containing the coordinates of the quantile
	\details This is equivalent to GetQuantile() but returns an std::vector intead of an Eigen::VectorXd
	*/
	std::vector<double> GetQuantileVector(double Prob)const;
#ifdef mvNormSamplerUnsafeMethods
	/** \name Unsafe Methods
	The methods in this group use unsafe memory access or return arrays allocated on the heap that must be manually deleted.

	These functions are normally not compiled for safety reasons. To use them, the mvNormSamplerUnsafeMethods symbol must be defined at compile time
	\{
	*/
	/**
	\brief Set the expected values vector
	\param mVect an array containing the new values for the expected values vector
	\details This is an overloaded version of SetMeanVector(const Eigen::VectorXd&)
	\warning This function will search for a number of elements equal to the dimensionality of the distribution in the array. This may mean accessing unallocated memory blocks if the supplied array is not big enough
	 */
	void SetMeanVector(double* mVect);
	/**
	\brief Set the Var-Cov matrix of the distribution
	\param mVect a matrix containing the new values for the variance covariance matrix of the distribution
	\details This is an overloaded version of SetVarCovMatrix(const Eigen::MatrixXd&)
	\warning This function will search for a number of elements, both in the row and the column dimension, equal to the dimensionality of the distribution in the matrix. This may mean accessing unallocated memory blocks if the supplied matrix is not big enough
	 */
	bool SetVarCovMatrix(double** mVect);
	/**
	\brief Generates a single simulation from the distribution
	\return A dynamically allocated array with number of elements equal to the dimensionality of the distribution representing a single extraction from the distribution
	\details This is equivalent to ExtractSample() but returns an array intead of an Eigen::RowVectorXd. If it can't be calculated, NULL is returned
	\warning This function will return an array allocated on the heap. If the user doesn't take care of deleting it, this will lead to memory leaks
	*/
	double* ExtractSampleArray() const;
	/**
	\brief Generates a single simulation from the distribution
	\param NumSamples The number of simulation to run
	\return A dynamically allocated matrix with number of columns equal to the dimensionality of the distribution and number of rows equal to the number of simulations representing multiple draws from the distribution
	\details This is equivalent to ExtractSamples() but returns a matrix intead of an Eigen::MatrixXd. If it can't be calculated, NULL is returned
	\warning This function will return a matrix allocated on the heap. If the user doesn't take care of deleting it, this will lead to memory leaks
	*/
	double** ExtractSamplesMatix(unsigned int NumSamples) const;
	/**
	\brief Generates a single simulation from the distribution and returns its marginal CDF
	\return A dynamically allocated array with number of elements equal to the dimensionality of the distribution representing a single extraction from the distribution
	\details This is equivalent to ExtractSampleCDF() but returns an array intead of an Eigen::RowVectorXd. If it can't be calculated, NULL is returned
	\warning This function will return an array allocated on the heap. If the user doesn't take care of deleting it, this will lead to memory leaks
	*/
	double* ExtractSampleCDFArray() const;
	/**
	\brief Generates a single simulation from the distribution and returns its marginal CDF
	\param NumSamples The number of simulation to run
	\return A dynamically allocated matrix with number of columns equal to the dimensionality of the distribution and number of rows equal to the number of simulations representing multiple draws from the distribution
	\details This is equivalent to ExtractSamplesCDF() but returns a matrix intead of an Eigen::MatrixXd. If it can't be calculated, NULL is returned
	\warning This function will return a matrix allocated on the heap. If the user doesn't take care of deleting it, this will lead to memory leaks
	*/
	double** ExtractSamplesCDFMatix(unsigned int NumSamples) const;
	/**
	\brief Computes the probability density function of the distribution in correspondence of the supplied coordinates
	\param Coordinates An array containing the coordinates of the point for which the pdf should be computed
	\param GetLogDensity If set to true the log density is returned instead of the actual density
	\return The value of the probability density function
	\details This is an overloaded version of GetDensity(const Eigen::VectorXd&, bool)
	\warning This function will search for a number of elements equal to the dimensionality of the distribution in the array. This may mean accessing unallocated memory blocks if the supplied array is not big enough
	*/
	double GetDensity(double* Coordinates, bool GetLogDensity=false)const;
	//! Computes the cumulative density function of the distribution in correspondence of the supplied coordinates
	/*!
	\param Coordinates A vector containing the coordinates of the point for which the cdf should be computed
	\param UseGenz If set to true the algorithm described in Genz (1992) to calculate the cdf, otherwise it will use full monte-carlo estimation (much slower)
	\param NumSimul The maximum number of simulations for the Genz algorithm. If UseGenz is false this is the number of simulations that will be run by monte-carlo
	\return The value of the cumulative density function
	\details This is an overloaded version of GetCumulativeDesity(const Eigen::VectorXd& Coordinates, bool UseGenz, unsigned int NumSimul)
	\warning This function will search for a number of elements equal to the dimensionality of the distribution in the array. This may mean accessing unallocated memory blocks if the supplied array is not big enough
	*/
	double GetCumulativeDesity(double* Coordinates, bool UseGenz=true, unsigned int NumSimul=500000)const;
	//! Computes the inverse cumulative density function of the distribution in correspondence of the supplied probability
	/*!
	\return A dynamically allocated array containing the coordinates of the quantile
	\details This is equivalent to GetQuantile() but returns an array intead of an Eigen::VectorXd. If it can't be calculated, NULL is returned
	\warning This function will return a matrix allocated on the heap. If the user doesn't take care of deleting it, this will lead to memory leaks
	*/
	double* GetQuantileArray(double Prob);
	/// \}
#endif
private:
	template <class F, class T> friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits);
	template <class F, class T>	friend T boost::math::tools::newton_raphson_iterate(F f, T guess, T min, T max, int digits, boost::uintmax_t& max_iter);
	template <class F, class T> friend void boost::math::tools::detail::handle_zero_derivative(F f,T& last_f0,const T& f0,T& delta,T& result,T& guess,const T& min,const T& max);
};
} //namespace Multivariate
#endif // NormalDist_h__
