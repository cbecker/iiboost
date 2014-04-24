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

#ifndef _EIGENUTILS_H_
#define _EIGENUTILS_H_

#include <cstdio>
#include <string>

/// writes a matrix to a file, in binary format
///  just for tests
template<typename T>
static void writeMatrix( const std::string &fName, const T &m )
{
	FILE *f = fopen( fName.c_str(), "wb" );
	
	if (f == NULL)
		qFatal("Could not open for writing: %s", fName.c_str());
		
	// get dimensions
    const unsigned int rows = m.rows();
    const unsigned int cols = m.cols();
	
	if ( 1 != fwrite( (const void *)&rows, sizeof(rows), 1, f ) )
		qFatal("Error writing file");
		
	if ( 1 != fwrite( (const void *)&cols, sizeof(cols), 1, f ) )
		qFatal("Error writing file");
		
	// now data
    if ( 1 != fwrite( (const void *)m.data(), ((unsigned long int)rows)*((unsigned long int)cols)*((unsigned long int)sizeof(*m.data())), 1, f ) )
		qFatal("Error writing file");
	
	fclose(f);
}

#endif
