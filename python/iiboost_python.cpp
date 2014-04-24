#include <stdexcept>
#include <cstdio>
#include <cstring>

// error handling (name comes from Qt)
#define qFatal(...) do { char errStr[8192]; sprintf(errStr, "ERROR: "); sprintf(errStr, __VA_ARGS__);sprintf(errStr, "\n"); \
 					     throw std::runtime_error(errStr); } while(0)
#define qDebug(...) do { fprintf (stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); } while(0)


#include "ROIData.h"
#include "BoosterInputData.h"
#include "ContextFeatures/ContextRelativePoses.h"

#include "Booster.h"
#include "utils/TimerRT.h"

#include <Python.h>

typedef float PredictionPixelType;

extern "C"
{
	void freeModel( void *modelPtr )
	{
		if (modelPtr == 0)	return;

		delete ((BoosterModel *)modelPtr);
	}

	// assumes that predPtr is already allocated, of same size as imgPtr
	void predict( void *modelPtr, ImagePixelType *imgPtr, int width, int height, int depth, PredictionPixelType *predPtr )
	{
		Matrix3D<PredictionPixelType> predMatrix;
		predMatrix.fromSharedData( predPtr, width, height, depth );

		// create roi for image, no GT available
		ROIData roi;
		roi.init( imgPtr, 0, 0, 0, width, height, depth );

		// raw image to integral image
		ROIData::IntegralImageType ii;
		ii.compute( roi.rawImage );
		roi.addII( ii.internalImage().data() );


		MultipleROIData allROIs;
		allROIs.add( &roi );

		Booster adaboost;
		adaboost.setModel( *((BoosterModel *) modelPtr) );

		adaboost.predict( &allROIs, &predMatrix );
	}

	// returns a BoosterModel *
	void * train( ImagePixelType *imgPtr, GTPixelType *gtPtr, 
				  int width, int height, int depth,
				  int numStumps, int debugOutput )
	{
		BoosterModel *modelPtr = 0;

		try
		{
			ROIData roi;
			roi.init( imgPtr, gtPtr, 0, 0, width, height, depth );

			roi.rawImage.save("/tmp/test.nrrd");

			// raw image to integral image
			ROIData::IntegralImageType ii;
			ii.compute( roi.rawImage );
			roi.addII( ii.internalImage().data() );

			//qFatal("Hey!");

			MultipleROIData allROIs;
			allROIs.add( &roi );

			BoosterInputData bdata;
			bdata.init( &allROIs );
			bdata.showInfo();

			Booster adaboost;
			adaboost.setShowDebugInfo( debugOutput != 0 );

			adaboost.train( bdata, numStumps );

			// create by copying
			modelPtr = new BoosterModel( adaboost.model() );
		}
		catch(std::exception const& e)
		{
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}

		return modelPtr;
	}
}
