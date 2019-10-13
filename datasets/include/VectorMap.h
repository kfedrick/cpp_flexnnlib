//
// Created by kfedrick on 10/12/19.
//

#ifndef FLEX_NEURALNET_VMAP_H_
#define FLEX_NEURALNET_VMAP_H_

#include <string>
#include <valarray>

namespace flexnnet
{
   class VectorMap
   {
   public:
      /**
       * Assign the value of the member vector named by '_name' to the value specified
       * by '_val'. The named vector must already exist and must be the same size as
       * the '_val'; otherwise an exception will be thrown.
       *
       * @param _name
       * @param _val
       */
      virtual void assign(std::string _name, const std::valarray<double>& _val) = 0;


      /**
       * Return a list with the names of the member vectors.
       *
       * @return
       */
      virtual const std::vector<std::string>& keyset(void) const = 0;

      /**
       * Return a const reference to the member vector named by '_name'.
       * @param _name
       * @return
       */
      virtual const std::valarray<double>& operator[](const std::string& _name) const = 0;

      /**
       * Return a const reference to the member vector named by '_name'.
       * @param _name
       * @return
       */
      virtual const std::valarray<double>& at(const std::string& _name) const = 0;

      /**
       * Concatenate all member vectors into a single vector and return the value.
       * @return
       */
      virtual const std::valarray<double>& operator()(void) const = 0;

   };
}

#endif //FLEX_NEURALNET_VMAP_H_
