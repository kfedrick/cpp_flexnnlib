//
// Created by kfedrick on 5/12/19.
//

#ifndef FLEX_NEURALNET_BASICNEURALNET_H_
#define FLEX_NEURALNET_BASICNEURALNET_H_

#include <map>
#include "NetworkLayer.h"
#include "NetworkOutput.h"
//#include "Datum.h"

namespace flexnnet
{
   class BasicNeuralNet : public NamedObject
   {

   public:
      BasicNeuralNet (const std::vector<std::shared_ptr<NetworkLayer>> &layers, bool _recurrent, const std::string &_name = "BasicNeuralNet");
      virtual ~BasicNeuralNet ();

   public:
      /**
       * BasicNeuralNet::size(void)
       *    Returns the size of the network output vector.
       *
       * @return - the size of the network output vector.
       */
      size_t size (void);

   public:

      /**
       * Initialize network weights and other network state information
       * prior to training.
       */
      virtual void initialize_weights (void);

      /**
       * Reset network state activation prior to presentation of next
       * exemplar or episode.
       */
      virtual void reset(void);

      /**
       * Activate the network with the specified input data
       *
       * @param _indatum - the input datum to the network
       * @return - the network output for the specified input data
       */
      virtual const Datum &activate (const Datum &_indatum);

   public:
      const std::set<std::string> &get_layer_names (void);

   public:
      std::string toJSON(void) const;
      const std::vector<std::shared_ptr<NetworkLayer>> get_layers(void) const;
      //const Datum& get_network_input(void) const;

   private:
      /**
       * Private helper function to initialize network_output_conn by adding connections
       * from to network_output_conn from all output layers.
       *
       * @return - the size of the virtual output vector.
       */
      std::map<std::string, std::valarray<double>> init_network_output_layer (void);

   private:
      size_t network_output_size;

      // Network layers stored in proper activation order
      std::vector<std::shared_ptr<NetworkLayer>> network_layers;

      // Set containing layer names
      std::set<std::string> layer_name_set;

      // recurrent_network_flag - Set if this network has recurrent connections.
      bool recurrent_network_flag;

      // network_output_conn - Used to coelesce network output from the output layers
      // and to scatter network backpropagation error to the output layers.
      //
      NetworkOutput network_output_conn;

      // network_output_pattern - Cached value for the most recent network activation
      //
      //Datum network_output_pattern;

      // network_input - Cached value for the most recent network input value
      //
      //Datum network_input;
   };

   inline const std::set<std::string> &BasicNeuralNet::get_layer_names (void)
   {
      return layer_name_set;
   }

   inline size_t BasicNeuralNet::size (void)
   {
      return network_output_conn.virtual_input_size ();
   }

   inline const std::vector<std::shared_ptr<NetworkLayer>> BasicNeuralNet::get_layers(void) const
   {
      return network_layers;
   }

   /*
   inline const Datum& BasicNeuralNet::get_network_input(void) const
   {
      return network_input;
   }
    */

}

#endif //FLEX_NEURALNET_BASICNEURALNET_H_
