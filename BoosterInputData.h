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

#ifndef _BOOSTER_INPUT_DATA_H_
#define _BOOSTER_INPUT_DATA_H_

#include "ROIData.h"
#include <Eigen/Dense>
#include "globaldefs.h"
#include <memory>


// contains image data, integral images, etc
struct MultipleROIData
{
	typedef std::shared_ptr<ROIData>	ROIDataPtr;

	// list of ROIs
	std::vector<ROIDataPtr>	ROIs;

	// anisotropy in Z
	float	zAnisotropyFactor;
	float	invZAnisotropyFactor;	// automatically updated with init()


	void init( float _zAnisotropyFact ) 
	{
		zAnisotropyFactor = _zAnisotropyFact;
		invZAnisotropyFactor = 1.0 / zAnisotropyFactor;
	}

	void clear()
	{
		ROIs.clear();
	}

	void add( ROIDataPtr roiPtr )
	{
		ROIs.push_back(roiPtr);
	}

	inline unsigned numROIs() const { return ROIs.size(); }

	MultipleROIData() { zAnisotropyFactor = invZAnisotropyFactor = 1.0; }
};


// input to the boosting algorithm
// contains info about all ROIs
struct BoosterInputData
{
	// to identify a 3D location
	//  we use float bcos it avoids cast while training/predicting
	typedef Eigen::Vector3f	LocType;

	typedef std::shared_ptr<MultipleROIData>	MultipleROIDataPtr;
	typedef std::shared_ptr<const MultipleROIData>	MultipleROIDataConstPtr;

	MultipleROIDataConstPtr 	imgData; // image data itself, containing many ROIs

	// now data for each sample
	std::vector<unsigned>		sampROI;		// which ROI it belongs to
	std::vector<GTPixelType>	sampLabels;		// its label (0->neg, anything else -> pos)

	std::vector<LocType> 		sampLocation;	// x,y,z location
	std::vector<unsigned>		sampOffset;		// offset in terms of the 3D image

	void clear()
	{
		sampROI.clear();
		sampLabels.clear();
		sampLocation.clear();
		sampOffset.clear();
	}

	void showInfo()
	{
		qDebug("--- BoosterInputData ---");
		qDebug("zAnisotropyFactor: %.4f", imgData->zAnisotropyFactor);
		qDebug("\tNum ROIs: %lu", imgData->ROIs.size());

		unsigned nPos = 0, nNeg = 0;
		for (unsigned i=0; i < sampLabels.size(); i++)
			if (sampLabels[i] == GTPosLabel)
				nPos++;
			else
				nNeg++;

		qDebug("\tNum pos: %u", nPos);
		qDebug("\tNum neg: %u", nNeg);
		qDebug("--- End BoosterInputData ---");
	}

	void init(  MultipleROIDataConstPtr rois,
				bool ignoreGT = false, 
				bool debugInfo = false,
				const int minBorderDist = 10 )
	{
		clear();
		imgData = rois;

		// z border ignore distance
		const int minBorderDistZ = std::min( (int)1, (int)ceil(minBorderDist/rois->zAnisotropyFactor) );

		for (unsigned curROIIdx=0; curROIIdx < imgData->numROIs(); curROIIdx++)
		{
			const MultipleROIData::ROIDataPtr &roi = imgData->ROIs[curROIIdx];

			// find out if GT is there
			const bool hasGT = (roi->gtImage.isEmpty() == false) && (!ignoreGT);


			unsigned numFound = 0;
			if (hasGT)
			{
				const Matrix3D<GTPixelType> &gt = roi->gtImage;
				const unsigned numVoxels = gt.numElem();

				const int maxX = gt.width() - minBorderDist;
				const int maxY = gt.height() - minBorderDist;
				const int maxZ = gt.depth() - minBorderDistZ;

				// go through the image, find pos/neg samples
				for (unsigned i=0; i < numVoxels; i++)
				{
					const GTPixelType label = gt.data()[i];

					if ( (label == GTPosLabel) || (label == GTNegLabel) )
					{

						{
							unsigned x,y,z;
							gt.idxToCoord(i, x, y, z);	// convert to coords

							if ( x < minBorderDist )	continue;
							if ( x > maxX )	continue;
							
							if ( y < minBorderDist )	continue;
							if ( y > maxY )	continue;

							if ( z < minBorderDistZ )	continue;
							if ( z > maxZ )	continue;

							sampLocation.push_back( LocType(x,y,z) );
						}

						sampLabels.push_back(label);
						sampOffset.push_back(i);

						numFound++;
					}
				}
			}
			else
			{
				const Matrix3D<GTPixelType> &raw = roi->rawImage;
				const unsigned numVoxels = raw.numElem();

				// go through the image
				for (unsigned i=0; i < numVoxels; i++)
				{
					sampOffset.push_back(i);

					{
						unsigned x,y,z;
						raw.idxToCoord(i, x, y, z);	// convert to coords

						sampLocation.push_back( LocType(x,y,z) );
					}

					numFound++;
				}
			}

			// add constant to sampROI
			sampROI.insert( sampROI.end(), numFound, curROIIdx );

			if (debugInfo)
				qDebug("Added ROI %u: %u samples", curROIIdx, numFound);
		}
	}
};

#endif
