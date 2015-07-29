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

// error handling (name comes from Qt)
#define qFatal(...) do { fprintf(stderr, "ERROR: "); fprintf (stderr, __VA_ARGS__); fprintf(stderr, "\n");  exit(-1); } while(0)
#define qDebug(...) do { fprintf (stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); } while(0)

#include "ROIData.h"
#include "BoosterInputData.h"
#include "ContextFeatures/ContextRelativePoses.h"

#include "Booster.h"
#include "utils/TimerRT.h"

int main()
{
	Matrix3D<ImagePixelType> img, gt;

	if (!img.load("../testData/single_synapse.tif"))
		qFatal("Error loading image");

	if (!gt.load("../testData/single_synapse_testgt.tif"))
		qFatal("Error loading image");


	ROIData roi;
    roi.init( img.data(), gt.data(), 0, 0, img.width(), img.height(), img.depth(), 1.0 );

    // --> VERY IMPORTANT, set ground truth labels
    roi.setGTNegativeSampleLabel( 1 );
    roi.setGTPositiveSampleLabel( 2 );

	// raw image integral image
	ROIData::IntegralImageType ii;
	ii.compute( img );
	roi.addII( ii.internalImage().data() );

	MultipleROIData allROIs;
	allROIs.add( shared_ptr_nodelete(ROIData, &roi) );

	BoosterInputData bdata;
	bdata.init( shared_ptr_nodelete(MultipleROIData, &allROIs) );
	bdata.showInfo();

	Booster adaboost;

	adaboost.train( bdata, 100 );

	// predict
	Matrix3D<float> predImg;
	
	TimerRT timer; 

	// ---- No early stopping
	timer.reset();
	adaboost.predict<false>( allROIs, &predImg );
	qDebug("Elapsed: %f", timer.elapsed());
	predImg.save("/tmp/test.nrrd");


	// ---- With early stopping
	timer.reset();
	adaboost.predict<true>( allROIs, &predImg );
	qDebug("Elapsed early stop: %f", timer.elapsed());
	predImg.save("/tmp/test-earlystop.nrrd");


	// --- now same tests, but with predict with double polarity
	// ---- No early stopping
	timer.reset();
	adaboost.predictDoublePolarity<false>( allROIs, &predImg );
	qDebug("Elapsed: %f", timer.elapsed());
	predImg.save("/tmp/test-2pol.nrrd");


	// ---- With early stopping
	timer.reset();
	adaboost.predictDoublePolarity<true>( allROIs, &predImg );
	qDebug("Elapsed early stop: %f", timer.elapsed());
	predImg.save("/tmp/test-2pol-earlystop.nrrd");

	// with early stopping + feature ordering
	timer.reset();
	adaboost.predictWithFeatureOrdering<true>( allROIs, &predImg );
	qDebug("Elapsed feat ordering: %f", timer.elapsed());
	predImg.save("/tmp/test-featord.nrrd");

	// with ROI for single slice + early stopping + feature ordering
	timer.reset();
	ROICoordinates subROI;
	subROI.x1 = subROI.y1 = 0;
	subROI.z1 = subROI.z2 = img.depth() / 2;
	subROI.x2 = img.width() - 1;
	subROI.y2 = img.height() - 1;

	subROI.printInfo();

	adaboost.predictWithFeatureOrdering<true>( allROIs, &predImg, 0, IIBOOST_NUM_THREADS, &subROI );
	qDebug("Elapsed ROI + feat ordering: %f", timer.elapsed());
	predImg.save("/tmp/test-roi-featord.nrrd");

	// save JSON model
	if (!adaboost.saveModelToFile( "/tmp/model.json" ))
		std::cout << "Error saving JSON model" << std::endl;

	return 0;
}
