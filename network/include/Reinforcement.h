//
// Created by kfedrick on 5/20/21.
//

#ifndef FLEX_NEURALNET_REINFORCEMENT_H_
#define FLEX_NEURALNET_REINFORCEMENT_H_

namespace flexnnet
{

   class Reinforcement
   {
   public:
      virtual size_t size() const = 0;

      virtual const double& operator[](size_t _ndx) const = 0;
      virtual const double& at(size_t _ndx) const = 0;
      virtual const double& at(const std::string& _field) const = 0;

      virtual const std::vector<std::string>& get_fields() const = 0;
      virtual const std::valarray<double>& value() const = 0;
   };

}

#endif //FLEX_NEURALNET_REINFORCEMENT_H_
