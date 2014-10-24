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

	if (!img.load("/cvlabdata1/cvlab/espina/sample_data/Madrid_Train.tif"))
		qFatal("Error loading image");

	if (!gt.load("/cvlabdata1/cvlab/espina/sample_data/Madrid_Train_espinagt.tif"))
		qFatal("Error loading image");


	ROIData roi;
	roi.init( img.data(), gt.data(), 0, 0, img.width(), img.height(), img.depth() );

	// raw image integral image
	ROIData::IntegralImageType ii;
	ii.compute( img );
	roi.addII( ii.internalImage().data() );

	MultipleROIData allROIs;
	allROIs.add( &roi );

	BoosterInputData bdata;
	bdata.init( &allROIs );
	bdata.showInfo();

	Booster adaboost;

	adaboost.train( bdata, 100 );

	// predict
	Matrix3D<float> predImg;
	TimerRT timer; timer.reset();
	adaboost.predict( &allROIs, &predImg );
	qDebug("Elapsed: %f", timer.elapsed());
	predImg.save("/tmp/test.nrrd");

	// save JSON model
	if (!adaboost.saveModelToFile( "/tmp/model.json" ))
		std::cout << "Error saving JSON model" << std::endl;

	return 0;
}
