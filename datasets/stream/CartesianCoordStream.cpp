//
// Created by kfedrick on 3/13/21.
//

#include <CartesianCoord.h>
#include "picojson.h"

std::ostream&
operator<<(std::ostream& _ostrm, const flexnnet::CartesianCoord& _coord)
{
   _ostrm << "{\"x\":" << _coord.x << ",\"y\":" << _coord.y << "}";
   return _ostrm;
}

std::istream&
operator>>(std::istream& _istrm, flexnnet::CartesianCoord& _coord)
{
   picojson::value picoval;

   // Read next json value
   _istrm >> picoval;

   // Get the picojson object, extract the elements and assign
   // them to the CartesianCoord object.
   const picojson::object& o = picoval.get<picojson::object>();
   _coord.x = o.at("x").get<double>();
   _coord.y = o.at("y").get<double>();

   // Return the remainder of the input stream
   return _istrm;
}

