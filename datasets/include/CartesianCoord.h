//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_CARTESIANCOORD_H_
#define FLEX_NEURALNET_CARTESIANCOORD_H_

#include <string>
#include <map>
#include <valarray>
#include <flexnnet.h>
#include <ValarrayMap.h>

// Forward declaration for CartesianCoord
namespace flexnnet { class CartesianCoord; }

// Forward declarations for stream operators
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::CartesianCoord& _coord);
std::istream& operator>>(std::istream& _istrm, flexnnet::CartesianCoord& _coord);

namespace flexnnet
{
   class CartesianCoord : public NNetIOInterface
   {
   public:
      CartesianCoord();
      CartesianCoord(double _x, double _y);

      virtual size_t size(void) const;

      const std::valarray<double>& value(void) const override;

      const flexnnet::ValarrMap& value_map(void) const override;

      void parse(const flexnnet::ValarrMap& _vmap) override;

      CartesianCoord& operator=(const CartesianCoord& _value);

      friend std::ostream& ::operator<<(std::ostream& _ostrm, const flexnnet::CartesianCoord& _coord);

      friend std::istream& ::operator>>(std::istream& _istrm, flexnnet::CartesianCoord& _coord);

   private:

      // x lower and upper extents
      std::pair<double, double> x_range;

      // x lower and upper extents
      std::pair<double, double> y_range;

      unsigned int N{11};
      double x;
      double y;
      mutable ValarrMap kernel_coord;
      mutable std::valarray<double> virtual_vector;
   };

   inline
   size_t CartesianCoord::size(void) const
   {
      return virtual_vector.size();
   }

} // end of flexnnet namespace


#endif //FLEX_NEURALNET_CARTESIANCOORD_H_
