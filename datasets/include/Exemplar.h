/*
 * Exemplar.h
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_EXEMPLAR_H_
#define FLEX_NEURALNET_EXEMPLAR_H_

namespace flexnnet
{

   template<class _InElem, class _TgtElem>
   class Exemplar
   {
   public:
      Exemplar ();
      Exemplar (const _InElem &in, const _TgtElem &out);

      virtual ~Exemplar ();

      const _InElem &input () const;
      const _TgtElem &target_output () const;

      void toFile (fstream &_fs);
      void fromFile (fstream &_fs);

   private:
      _InElem network_input;
      _TgtElem target_network_output;
   };

   template<class _InElem, class _TgtElem>
   Exemplar<_InElem, _TgtElem>::Exemplar ()
   {
   }

   template<class _InElem, class _TgtElem>
   Exemplar<_InElem, _TgtElem>::Exemplar (const _InElem &in, const _TgtElem &out)
   {
      network_input = in;
      target_network_output = out;
   }

   template<class _InElem, class _TgtElem>
   Exemplar<_InElem, _TgtElem>::~Exemplar ()
   {
   }

   template<class _InElem, class _TgtElem>
   const _InElem &Exemplar<_InElem, _TgtElem>::input () const
   {
      return network_input;
   }

   template<class _InElem, class _TgtElem>
   const _TgtElem &Exemplar<_InElem, _TgtElem>::target_output () const
   {
      return target_network_output;
   }

   template<class _InElem, class _TgtElem>
   void Exemplar<_InElem, _TgtElem>::toFile (fstream &_fs)
   {
      network_input.toFile (_fs);
      target_network_output.toFile (_fs);
   }

   template<class _InElem, class _TgtElem>
   void Exemplar<_InElem, _TgtElem>::fromFile (fstream &_fs)
   {
      network_input.fromFile (_fs);
      target_network_output.fromFile (_fs);
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_EXEMPLAR_H_ */
