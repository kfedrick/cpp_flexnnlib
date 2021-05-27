//
// Created by kfedrick on 3/12/21.
//

#ifndef FLEX_NEURALNET_JSONSTREAM_H_
#define FLEX_NEURALNET_JSONSTREAM_H_

#include <ostream>
#include <sstream>
#include <valarray>

namespace flexnnet
{
   class JSONStream
   {
   public:

      /**
       * Return a JSON encoded key/vectorize pair.
       *
       * @tparam _Typ
       * @param _key
       * @param _val
       * @param _indent
       * @return
       */
      template<typename _Typ>
      static
      std::string
      tojson(const std::string& _key, const _Typ& _val, size_t _indent = 0)
      {
         std::ostringstream ss;

         ss << "\"" << _key.c_str() << "\":";
         ss << tojson(_val);

         return ss.str();
      }

      template<typename _Typ>
      static
      std::string
      tojson(const _Typ& _val, size_t _indent = 0)
      {
         std::ostringstream ss;
         ss << _val;
         return ss.str();
      }

      /**
       * Return a json encoded string vectorize.
       *
       * @param _val
       * @param _indent
       * @return
       */
      static std::string
      tojson(const std::string& _val, size_t _indent = 0);

      /**
       * Return a json encoded array of double.
       *
       * @param _valarr
       * @param _indent
       * @return
       */
      static std::string
      tojson(const std::valarray<double>& _valarr, size_t _indent = 0);

      /**
       * Return a json encoded array of _Typ.
       *
       * @tparam _Typ
       * @param _vec
       * @param _indent
       * @return
       */
      template<typename _Typ>
      static std::string
      tojson(const std::vector<_Typ>& _vec, size_t _indent = 0);
   };
}

#endif //FLEX_NEURALNET_JSONSTREAM_H_
