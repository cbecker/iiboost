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

#ifndef __BOSTER_MODEL_H_
#define __BOSTER_MODEL_H_

/**
 * This file contains booster weak learner/weight class (BoosterComponent)
 * and the full boosting model (BoosterModel), which is basically an array of BoosterComponent
 */

// for serialization
#include <rapidjson/document.h>
#include <rapidjson/filestream.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>


// contains weak learner and weight
struct BoosterComponent
{
	WeakLearner wl;
	double		alpha;

	BoosterComponent( const WeakLearner &_wl, const double _alpha ) :
		wl(_wl), alpha(_alpha) {}

	BoosterComponent() :
		alpha(0.0) {}

	// JSON serialization, requires passing the allocator bcos of rapidjson
    void serialize( rapidjson::Value &obj, 
                    			rapidjson::Document::AllocatorType& allocator ) const
    {
        obj.SetObject();

        obj.AddMember("alpha", alpha, allocator);

        rapidjson::Value jsonWL;
        wl.serialize(jsonWL, allocator);
        obj.AddMember("weaklearner", jsonWL, allocator);
    }

    // JSON deserialization
    bool deserialize( const rapidjson::Value &obj )
    {
        if (!obj.IsObject())
            return false;

        alpha = obj["alpha"].GetDouble();
        wl.deserialize( obj["weaklearner"] );
    }
};



// a collection of BoosterComponent
struct BoosterModel
{
	std::vector<BoosterComponent> data;
	std::string mIIBoostModelVersion;

	BoosterModel()
	{
		mIIBoostModelVersion = "IIBoost Model v1.0";
	}

	void add( const WeakLearner &wl, const double alpha )
	{
		data.push_back( BoosterComponent(wl, alpha) );
	}

	void clear() { data.clear(); }
	unsigned size() const { return data.size(); }

	inline const BoosterComponent &operator [](unsigned idx) const
	{
		return data[idx];
	}

	// save model with JSON
	//  for more handy functionality see
	//		serializeToFile()
	//		serializeToString()
	template<typename StreamType>
	void serialize( StreamType &stream ) const
	{
		rapidjson::Document doc;
		doc.SetObject();

		#define JSON_ADD( name, what ) \
		do { \
			rapidjson::Value	val(what); \
			doc.AddMember( name, val, doc.GetAllocator() ); \
		} while(0)

		rapidjson::PrettyWriter<StreamType>	writer(stream);
		JSON_ADD( "type", mIIBoostModelVersion.c_str() );

		rapidjson::Value  model;
		model.SetArray();
		model.Reserve(data.size(), doc.GetAllocator());

		for (unsigned i=0; i < data.size(); i++)
		{
			rapidjson::Value  v;
			data[i].serialize( v, doc.GetAllocator() );

			model.PushBack(v, doc.GetAllocator());
		}

		doc.AddMember( "model", model, doc.GetAllocator() );
		doc.Accept(writer);

		#undef JSON_ADD
	}

	// serializes to destStr
	void serializeToString( std::string *destStr ) const
	{
		rapidjson::StringBuffer strBuf;
		serialize(strBuf);
		*destStr = strBuf.GetString();
	}

	bool serializeToFile( const std::string &fName ) const
	{
		FILE *f = fopen(fName.c_str(), "w");
		if (f == 0)
			return false;

		try{
			rapidjson::FileStream fStream(f);
			serialize(fStream);
		} catch( std::exception& e )
		{
			fclose(f);
			return false;
		}

		fclose(f);
		return true;
	}

	// load model from JSON document
	bool deserialize( rapidjson::Document &doc )
	{
		if (!doc.IsObject())	return false;
		if ( mIIBoostModelVersion.compare(doc["type"].GetString()) != 0 )
			return false;

		const rapidjson::Value &modelArray = doc["model"];
		if (!modelArray.IsArray())	return false;

		const unsigned N = modelArray.Size();
		
		data.resize(N);

		for (unsigned i=0; i < N; i++)
			data[i].deserialize( modelArray[i] );
		return true;
	}

	bool deserializeFromFile( const std::string &fName )
	{
		FILE *f = fopen(fName.c_str(), "r");
		if (f == 0)
			return false;

		bool retVal = true;
		try{
			rapidjson::FileStream fStream(f);
			
			rapidjson::Document doc;
			doc.ParseStream<0>( fStream );

			retVal = deserialize( doc );
		} catch( std::exception& e )
		{
			fclose(f);
			return false;
		}

		fclose(f);
		return retVal;
	}


	bool deserializeFromString( const std::string &data )
	{
		rapidjson::Document doc;
		doc.Parse<0>( data.c_str() );

		return deserialize( doc );
	}
};

#endif /** __BOOSTER_MODEL_H_ **/
