//
// Created by kfedrick on 3/7/21.
//
#include "CartesianCoord.h"

using flexnnet::CartesianCoord;

CartesianCoord::CartesianCoord()
{
   x = 0;
   y = 0;

   kernel_coord["x"] = std::valarray<double>(N);
   kernel_coord["y"] = std::valarray<double>(N);

   virtual_vector.resize(1);
}

CartesianCoord::CartesianCoord(double _x, double _y)
{
   x = _x;
   y = _y;

   kernel_coord["x"] = std::valarray<double>(N);
   kernel_coord["y"] = std::valarray<double>(N);

   virtual_vector.resize(1);
   virtual_vector[0] = _x;
}

const std::valarray<double>& CartesianCoord::value(void) const
{
   return virtual_vector;
}

const flexnnet::ValarrMap&
CartesianCoord::value_map(void) const
{
   kernel_coord["x"] = 0;
   kernel_coord["x"][(x + N / 2)] = 1;

   kernel_coord["y"] = 0;
   kernel_coord["y"][(y + N / 2)] = 1;

   return kernel_coord;
}

void
CartesianCoord::parse(const flexnnet::ValarrMap& _vmap)
{
}

CartesianCoord&
CartesianCoord::operator=(const CartesianCoord& _value)
{
   N = _value.N;

   x = _value.x;
   y = _value.y;

   kernel_coord = _value.kernel_coord;

   return *this;
}



//namespace flexnnet
//{



/*   {
      char khar;
      std::string err = "Unexpected EOF!";

      // Read to left brace
      _istrm.get(khar);
      while (khar != '{' && khar != EOF)
      {
         _istrm.get(khar);
         //std::cout << "read coord 1b '" << khar << "'\n" << std::flush;
      }
      if (khar == EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != '"' && khar != EOF)
         _istrm.get(khar);
      if (khar == EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != 'x' && khar != EOF)
         _istrm.get(khar);
      if (khar == EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != '"' && khar != EOF)
         _istrm.get(khar);
      if (khar == EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != ':' && khar != EOF)
         _istrm.get(khar);
      if (khar == EOF)
         throw std::invalid_argument(err);

      // Read x coordinate
      _istrm >> _coord.x;

      // Read to comma
      _istrm.get(khar);
      while (khar != ',' && khar != EOF)
         _istrm.get(khar);
      if (khar == !EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != '"' && khar != EOF)
         _istrm.get(khar);
      if (khar == !EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != 'y' && khar != EOF)
         _istrm.get(khar);
      if (khar == !EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != '"' && khar != EOF)
         _istrm.get(khar);
      if (khar == !EOF)
         throw std::invalid_argument(err);

      _istrm.get(khar);
      while (khar != ':' && khar != EOF)
         _istrm.get(khar);
      if (khar == !EOF)
         throw std::invalid_argument(err);

      // Read y coordinate
      _istrm >> _coord.y;

      // Read to closing right brace
      _istrm.get(khar);
      while (khar != '}' && khar != EOF)
         _istrm.get(khar);
      if (khar == !EOF)
         throw std::invalid_argument(err);

      return _istrm;
   }*/
//}
