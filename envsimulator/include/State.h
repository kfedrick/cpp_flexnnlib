//
// Created by kfedrick on 5/21/21.
//

#ifndef FLEX_NEURALNET_STATE_H_
#define FLEX_NEURALNET_STATE_H_

namespace flexnnet
{
   template<size_t DEPTH, typename ...>
   class State;

   template<size_t DEPTH, typename F, typename ...Fs>
   class State<DEPTH, F,Fs...> : public State<DEPTH+1, Fs...>
   {
   public:
      State()
      {
         std::cout << "depth = " << DEPTH << "\n";
      };

   private:
      size_t depth{DEPTH};
      F data;
   };

   template<size_t DEPTH>
   class State<DEPTH>
   {
   public:
      State() {};
   };

}

#endif //_STATE_H_
