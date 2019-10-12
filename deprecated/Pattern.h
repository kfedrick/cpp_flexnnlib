/*
 * Pattern.h
 *
 *  Created on: Feb 4, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_PATTERN_H_
#define FLEX_NEURALNET_PATTERN_H_

#include <vector>
#include <fstream>

namespace flexnnet
{

   class Pattern
   {
   public:
      Pattern(int sz = 0);
      Pattern(const std::vector<unsigned int>& vsizes);
      Pattern(const std::vector<std::vector<double> >& vec);
      Pattern(const std::vector<double>& vec);
      ~Pattern();

      int size() const;

      const std::vector<double>& operator()(void) const;
      const std::vector<double>& at(int index) const;
      const std::vector<double>& operator[](int index) const;

      void operator=(const Pattern& patt);
      void operator=(const std::vector<std::vector<double> >& vec);
      void operator=(const std::vector<double>& vec);

      void push_back(const std::vector<double>& vec);

      void clear();

      operator const std::vector<double>&() const;

      void toFile(std::fstream& _fs);
      void fromFile(std::fstream& _fs);

   private:
      void coalesce() const;

   private:
      std::vector<std::vector<double> > data;
      mutable std::vector<double> coalesced_data;

      // Flag to indicate that the DataVectors have been coalesced
      mutable bool coalesced_flag;
   };

}

#endif /* FLEX_NEURALNET_PATTERN_H_ */
