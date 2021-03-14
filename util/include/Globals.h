//
// Created by kfedrick on 3/12/21.
//

#ifndef FLEX_NEURALNET_GLOBALS_H_
#define FLEX_NEURALNET_GLOBALS_H_

#include <string>
#include <typeinfo>

namespace flexnnet
{
   /**
    * Demangle type id, _name, returned by 'typeid' command.
    * @param name
    * @return
    */
   std::string demangle(const std::string& _name);

   /**
    * Return demangled type id for template type _Typ.
    *
    * @tparam _Typ
    * @return
    */
   template <typename _Typ>
   std::string type_id(void)
   {
      return demangle(typeid(_Typ).name());
   }
}
#endif //FLEX_NEURALNET_GLOBALS_H_
