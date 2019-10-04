/*
 * Exemplar.h
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_EXEMPLAR_H_
#define FLEX_NEURALNET_EXEMPLAR_H_

#include <fstream>

namespace flexnnet
{

   template<class _InElem, class _OutElem>
   class Exemplar
   {
   public:
      Exemplar ();
      Exemplar (const _InElem &in, const _OutElem &target);

      virtual ~Exemplar ();

      const _InElem &input () const;
      const _OutElem &target () const;

      bool operator<(const Exemplar<_InElem, _OutElem>& _other) const;

      void toFile (std::fstream &_fs);
      void fromFile (std::fstream &_fs);

   private:
      _InElem network_input;
      _OutElem network_output;
   };

   template<class _InElem, class _OutElem>
   Exemplar<_InElem, _OutElem>::Exemplar ()
   {
   }

   template<class _InElem, class _OutElem>
   Exemplar<_InElem, _OutElem>::Exemplar (const _InElem &in, const _OutElem &out)
   {
      network_input = in;
      network_output = out;
   }

   template<class _InElem, class _OutElem>
   Exemplar<_InElem, _OutElem>::~Exemplar ()
   {
   }

   template<class _InElem, class _OutElem>
   const _InElem &Exemplar<_InElem, _OutElem>::input () const
   {
      return network_input;
   }

   template<class _InElem, class _OutElem>
   const _OutElem &Exemplar<_InElem, _OutElem>::target () const
   {
      return network_output;
   }

   template<class _InElem, class _OutElem>
   void Exemplar<_InElem, _OutElem>::toFile (std::fstream &_fs)
   {
      network_input.toFile (_fs);
      network_output.toFile (_fs);
   }

   template<class _InElem, class _OutElem>
   void Exemplar<_InElem, _OutElem>::fromFile (std::fstream &_fs)
   {
      network_input.fromFile (_fs);
      network_output.fromFile (_fs);
   }

   template<class _InElem, class _OutElem>
   bool Exemplar<_InElem, _OutElem>::operator<(const Exemplar<_InElem, _OutElem>& _other) const
   {
      return (network_input.hashval() + network_output.hashval() < _other.network_input.hashval() + _other.network_output.hashval());
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_EXEMPLAR_H_ */
