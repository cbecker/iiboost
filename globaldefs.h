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

#ifndef _GLOBAL_DEFS_H_
#define _GLOBAL_DEFS_H_

/** Important Types **/
typedef double 			IntegralImagePixelType;  // this can make a huge difference in mem usage
typedef unsigned char	ImagePixelType;
typedef double AdaBoostErrorType;

// Ground Truth
typedef unsigned char	GTPixelType;
static const GTPixelType GTNegLabel = 1;
static const GTPixelType GTPosLabel = 2;

// supervoxel or voxel-based?
#define USE_SUPERVOXELS	1


#endif