//////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013 Carlos Becker                                             //
// Ecole Polytechnique Federale de Lausanne                                     //
// Contact <carlos.becker@epfl.ch> for comments & bug reports                   //
//                                                                              //
// This program is free software: you can redistribute it and/or modify         //
// it under the terms of the version 3 of the GNU General Public License        //
// as published by the Free Software Foundation.                                //
//                                                                              //
// This program is distributed in the hope that it will be useful, but          //
// WITHOUT ANY WARRANTY; without even the implied warranty of                   //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             //
// General Public License for more details.                                     //
//                                                                              //
// You should have received a copy of the GNU General Public License            //
// along with this program. If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////

#ifndef __BOOSTER_H_
#define __BOOSTER_H_

#include "BoosterInputData.h"
#include "ContextFeatures/WeakLearner.h"
#include "ContextFeatures/ContextRelativePoses.h"
#include "utils/DiscreteRandomSampler.h"
#include <omp.h>

// we need the boosting model
#include "BoosterModel.h"

// an operator to speed up prediction
template<typename PredType>
struct BoosterPredictOperator
{
	typedef typename PredType::Scalar ScalarType;
	
	PredType &mPred;
	const ScalarType mAlpha;

	inline BoosterPredictOperator( PredType &pred, ScalarType alpha ) :
		mPred(pred), mAlpha(alpha) {}

	inline void operator ()( const unsigned i, const bool what )
	{
		#pragma omp atomic
		mPred.coeffRef(i) += what ? mAlpha : -mAlpha;
	}
};


/**
 * Main boosting class
 */
class Booster
{
public:
	typedef Eigen::ArrayXd  WeightsArrayType;

private:
	// contains pose of each pose idx + possible box sizes
	ContextRelativePoses  mPoses;

	// -- OPTIONS begin
	unsigned	mWeakLearnersPerIter;	// num weak learners searched per iter
	unsigned	mNumToSubsamplePerClass;
	bool 		mLinesearchOnWholeData;	// if true, search for alpha on whole data (default)
										//  otherwise, do it on subsampled set
	bool 		mShowDebugInfo;
	// -- OPTIONS end

	BoosterModel	mModel;

	DiscreteRandomSampler<WeightsArrayType> mSampler;

public:
	Booster()
	{
		mWeakLearnersPerIter = 500;
		mNumToSubsamplePerClass = 500;
		mLinesearchOnWholeData = true;
		mShowDebugInfo = false;
	}

	void setShowDebugInfo(bool yes) { mShowDebugInfo = yes; }

	const BoosterModel &model() const { return mModel; }
	void setModel( const BoosterModel &model )
	{
		mModel = model;
	}

	/// BEGIN SAVE-LOAD FUNCTIONS ///
	bool saveModelToFile( const std::string &fName ) const
	{	return mModel.serializeToFile(fName); }

	bool loadModelFromFile( const std::string &fName )
	{	return mModel.deserializeFromFile(fName); }

	void saveModelToString( std::string *destStr ) const
	{	mModel.serializeToString( destStr ); }

	bool loadModelFromString( const std::string &data )
	{	return mModel.deserializeFromString(data); }
	/// END SAVE-LOAD FUNCTIONS ///


	void doWeightedResamplingPerClass( const WeightsArrayType &weights, 
							   const BoosterInputData &bid, 
							   BoosterInputData *subsampledBID,
							   WeightsArrayType *subsampledWeights )
	{
		// cache neg/pos idx, according to size of bid
		static unsigned cachedSize = 0;
		static std::vector<unsigned> posIdx, negIdx;

		if (cachedSize != bid.sampLabels.size())
		{
			const unsigned N = bid.sampLabels.size();

			// re-compute cache
			posIdx.clear(); negIdx.clear();


			for (unsigned i=0; i < N; i++)
				if ( bid.sampLabels[i] == GTPosLabel )
					posIdx.push_back(i);
				else
					negIdx.push_back(i);

			cachedSize = N;
			qDebug("Rand sample, numPos: %lu, numNeg: %lu", posIdx.size(), negIdx.size());
		}

		// extract weights
		const unsigned NP = posIdx.size();
		const unsigned NN = negIdx.size();

		WeightsArrayType wPos(NP), wNeg(NN);

		for (unsigned i=0; i < NP; i++)
			wPos.coeffRef(i) = weights.coeff( posIdx[i] );

		for (unsigned i=0; i < NN; i++)
			wNeg.coeffRef(i) = weights.coeff( negIdx[i] );

		const double wPosSum = wPos.sum();
		const double wNegSum = wNeg.sum();

		const double wEachPos = wPosSum / mNumToSubsamplePerClass;
		const double wEachNeg = wNegSum / mNumToSubsamplePerClass;
		

		// subsampled idxs, and subsample!
		std::vector<unsigned> ssPos, ssNeg;

		mSampler.sampleWithReplacement( wPos, ssPos, mNumToSubsamplePerClass );
		mSampler.sampleWithReplacement( wNeg, ssNeg, mNumToSubsamplePerClass );

		// get total subsampled (in the future, this can speed up training)
		const unsigned nsPos = ssPos.size();
		const unsigned nsNeg = ssNeg.size();
		const unsigned totSampled = nsPos + nsNeg;

		// re-assign
		subsampledBID->sampROI.resize( totSampled );
		subsampledBID->sampLabels.resize( totSampled );
		subsampledBID->sampLocation.resize( totSampled );
		subsampledBID->sampOffset.resize( totSampled );
		subsampledWeights->resize( totSampled );

		// -- Positives
		for (unsigned i=0; i < nsPos; i++)
		{
			const unsigned idx = posIdx[ssPos[i]];

			subsampledBID->sampROI[i] = bid.sampROI[idx];
			subsampledBID->sampLabels[i] = bid.sampLabels[idx];
			subsampledBID->sampLocation[i] = bid.sampLocation[idx];
			subsampledBID->sampOffset[i] = bid.sampOffset[idx];
		}
		subsampledWeights->head(nsPos).setConstant(wEachPos);

		// -- Negatives
		for (unsigned i=0; i < nsNeg; i++)
		{
			const unsigned idx = negIdx[ssNeg[i]];

			subsampledBID->sampROI[nsPos + i] = bid.sampROI[idx];
			subsampledBID->sampLabels[nsPos + i] = bid.sampLabels[idx];
			subsampledBID->sampLocation[nsPos + i] = bid.sampLocation[idx];
			subsampledBID->sampOffset[nsPos + i] = bid.sampOffset[idx];
		}
		subsampledWeights->tail(nsNeg).setConstant(wEachNeg);

		unsigned pos = 0;
		unsigned neg = 0;
		for (unsigned i=0; i < subsampledWeights->size(); i++)
			if (subsampledBID->sampLabels[i] == GTPosLabel)
				pos++;
			else
				neg++;

		// make sure to return normalized weights
		// TODO: comment this line out, not necessary
		*subsampledWeights *= 1.0 / subsampledWeights->sum();
	}

	void doWeightedResampling( const WeightsArrayType &weights, 
							   const BoosterInputData &bid, 
							   BoosterInputData *subsampledBID,
							   WeightsArrayType *subsampledWeights )
	{
		// subsampled idxs, and subsample!
		std::vector<unsigned> ss;

		mSampler.sampleWithReplacement( weights, ss, 2 * mNumToSubsamplePerClass );

		// get total subsampled (in the future, this can speed up training)
		const unsigned totSampled = ss.size();

		// re-assign
		subsampledBID->sampROI.resize( totSampled );
		subsampledBID->sampLabels.resize( totSampled );
		subsampledBID->sampLocation.resize( totSampled );
		subsampledBID->sampOffset.resize( totSampled );
		subsampledWeights->resize( totSampled );

		for (unsigned i=0; i < totSampled; i++)
		{
			const unsigned idx = ss[i];

			subsampledBID->sampROI[i] = bid.sampROI[idx];
			subsampledBID->sampLabels[i] = bid.sampLabels[idx];
			subsampledBID->sampLocation[i] = bid.sampLocation[idx];
			subsampledBID->sampOffset[i] = bid.sampOffset[idx];
		}
		subsampledWeights->setConstant(1.0 / totSampled);
	}

	void train( const BoosterInputData &bid, unsigned numIters, unsigned numThreads = omp_get_max_threads() )
	{
		// num samples
		const unsigned N = bid.sampLabels.size();

		// count pos/neg
		unsigned nPos = 0, nNeg = 0;
		for (unsigned i=0; i < N; i++)
			if (bid.sampLabels[i] == GTPosLabel)
				nPos++;
			else
				nNeg++;

		// balance loss
		const double posWeight = 1.0 / nPos;
		const double negWeight = 1.0 / nNeg;

		WeightsArrayType  weights(N);
		for (unsigned i=0; i < N; i++)
			if (bid.sampLabels[i] == GTPosLabel)
				weights.coeffRef(i) = posWeight;
			else
				weights.coeffRef(i) = negWeight;

		Eigen::ArrayXf weakPred;


		for (unsigned iter=0; iter < numIters; iter++)
		{
			// normalize weights
			weights *= 1.0 / weights.sum();

			/// Sub-sample data to learn weak learner
			BoosterInputData 	subsampledBID;
			WeightsArrayType 	subsampledWeights;

			subsampledBID.imgData = bid.imgData;

			//doWeightedResampling( weights, bid, &subsampledBID, &subsampledWeights );
			doWeightedResamplingPerClass( weights, bid, &subsampledBID, &subsampledWeights );

			WeakLearner newWL;

			// Learn weak learner
			double weakErr = WeakLearnerSearcher( mPoses, subsampledBID, subsampledWeights, mWeakLearnersPerIter ).learn( newWL );

			if (mShowDebugInfo)
				qDebug("Weak learner %d: %f, %s", iter, weakErr, newWL.getStringDescription().c_str());

			// predict weak learner on all data
			newWL.classifyMultipleROIs( mPoses, bid, weakPred, numThreads );


			// search for weight (alpha/linesearch)
			double alpha = 0;

			if (mLinesearchOnWholeData)
			{
				double wholeSetWeakErr = 0.0;
				for (unsigned q=0; q < N; q++)
					if ( (weakPred.coeffRef(q) > 0) != (bid.sampLabels[q] == GTPosLabel) )
						wholeSetWeakErr += weights.coeffRef(q);

				if (mShowDebugInfo)
					qDebug("WL err on whole dataset: %f", wholeSetWeakErr);

				alpha = 0.5 * log( (1.0 - wholeSetWeakErr) / wholeSetWeakErr );
			}
			else
			{
				// learn on subset
				if (weakErr < 1e-16)	// in case we get perfect classification (unlikely)
					weakErr = 1e-16;

				alpha = 0.5 * log( (1.0 - weakErr) / weakErr );
			}
			if (mShowDebugInfo)
				qDebug("Alpha: %f", alpha);


			// re-weight data
			const double expP = exp(alpha);
			const double expN = exp(-alpha);

			for (unsigned i=0; i < N; i++)
				if ( (weakPred.coeffRef(i) > 0) == (bid.sampLabels[i] == GTPosLabel) )
					weights.coeffRef(i) *= expN;
				else
					weights.coeffRef(i) *= expP;

			mModel.add( newWL, alpha );
		}
	}

	// predicts roiNo in rois
	void predict( const MultipleROIData *rois, 
				  Matrix3D<float> *pred,
				  unsigned roiNo = 0,
				  unsigned numThreads = omp_get_max_threads() ) const
	{
		MultipleROIData singleRoiData;
		singleRoiData.init( rois->zAnisotropyFactor );
		singleRoiData.add( rois->ROIs[roiNo] );

		BoosterInputData bd;
		bd.init(&singleRoiData, true);

		const unsigned N = mModel.size();
		Eigen::ArrayXf weakPred;
		
		pred->reallocSizeLike( singleRoiData.ROIs[0]->rawImage );
		pred->fill(0);

		typedef Eigen::Map< Eigen::ArrayXf, Eigen::Unaligned > 	MapType;
		MapType predMap( pred->data(), pred->numElem() );

		#pragma omp parallel for
		for (unsigned i=0; i < N; i++)
		{
			BoosterPredictOperator<MapType> op(predMap, mModel[i].alpha);
			mModel[i].wl.classifySingleROIWithOp( mPoses, bd, op, numThreads );
		}
	}
};

#endif