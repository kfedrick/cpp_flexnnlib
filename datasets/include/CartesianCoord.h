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
   class CartesianCoord : public NNetIOValue
   {
   public:
      CartesianCoord();
      CartesianCoord(double _x, double _y);

      /**
       * Return the coordinate encoded as a NNetIO_Map for use as an
       * input to a neural network.
       *
       * @return
       */
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
      mutable ValarrayMap kernel_coord;
   };
} // end of flexnnet namespace


#endif //FLEX_NEURALNET_CARTESIANCOORD_H_
