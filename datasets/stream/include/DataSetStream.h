//
// Created by kfedrick on 3/13/21.
//

#ifndef FLEX_NEURALNET_ENUMERATEDDATASETSTREAM_H_
#define FLEX_NEURALNET_ENUMERATEDDATASETSTREAM_H_

#include <Globals.h>
#include <JSONStream.h>
#include <cxxabi.h>
#include "picojson.h"

template<class _InTyp, class _OutTyp>
std::ostream&
operator<<(std::ostream& _ostrm, const flexnnet::DataSet<_InTyp,
                                                         _OutTyp>& _dataset)
{
   // Write json object opening parenthesis and header containing
   // the input and output data types.
   _ostrm << "{\n";

   std::string intyp = flexnnet::type_id<_InTyp>();
   std::string outtyp = flexnnet::type_id<_OutTyp>();

   _ostrm << "  " << flexnnet::JSONStream::tojson("input_type", intyp).c_str();
   _ostrm << ",\n";
   _ostrm << "  " << flexnnet::JSONStream::tojson("output_type", outtyp).c_str();
   _ostrm << ",\n";

   _ostrm << "  \"exemplars\":\n  [\n";

   unsigned int count = 0;
   for (auto& exemplar : _dataset)
   {
      if (count > 0)
         _ostrm << ",\n";

      _ostrm << exemplar;
      count++;
   }
   _ostrm << "\n  ]\n";

   // Close the root object and close the file.
   _ostrm << "}\n";

   return _ostrm;
}

template<class _InTyp, class _OutTyp>
std::ostream&
operator<<(std::ostream& _ostrm, const flexnnet::Exemplar<_InTyp, _OutTyp>& _exemplar)
{
   _ostrm << "    {\n";
   _ostrm << "      " << flexnnet::JSONStream::tojson("input", _exemplar.first);
   _ostrm << ",\n";
   _ostrm << "      " << flexnnet::JSONStream::tojson("target", _exemplar.second);
   _ostrm << "\n";
   _ostrm << "    }";
}

template<class _InTyp, class _OutTyp>
std::istream&
operator>>(std::istream& _istrm, flexnnet::Exemplar<_InTyp, _OutTyp>& _exemplar)
{
   picojson::value v;

   // Try to read the exemplar
   _istrm >> v;

   // Try to read the input type
   _istrm >> v;
   if (v.is<picojson::object>())
   {
      const picojson::object& o = v.get<picojson::object>();

      const picojson::value& in = o.at("input");
      flexnnet::CartesianCoord icoord;
      std::stringstream iss(in.serialize());
      iss >> icoord;

      const picojson::value& tgt = o.at("target");
      flexnnet::CartesianCoord ocoord;
      std::stringstream oss(tgt.serialize());
      oss >> ocoord;

      _exemplar = flexnnet::Exemplar<flexnnet::CartesianCoord,
                                     flexnnet::CartesianCoord>(icoord, ocoord);
   }

   return _istrm;
}

template<class _InTyp, class _OutTyp>
std::istream&
operator>>(std::istream& _istrm, flexnnet::DataSet<_InTyp, _OutTyp>& _dataset)
{
   picojson::value v;

   // Try to read the input type
   _istrm >> v;
   if (v.is<picojson::object>())
   {
      const picojson::object& o = v.get<picojson::object>();

      unsigned int i = 0;
      picojson::array exemplars = o.at("exemplars").get<picojson::array>();
      for (picojson::array::const_iterator it = exemplars.begin(); it != exemplars.end();
           ++it)
      {
         std::stringstream ss(it->serialize());

         flexnnet::Exemplar<flexnnet::CartesianCoord, flexnnet::CartesianCoord> x;
         ss >> x;

         //flexnnet::CartesianCoord cc = *i;
         _dataset.push_back(x);
      }
   }

   return _istrm;
}

#endif //FLEX_NEURALNET_ENUMERATEDDATASETSTREAM_H_
