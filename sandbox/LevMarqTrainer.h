/*
 * LevMarqTrainer.h
 *
 *  Created on: Feb 26, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_BaseNeuralNet_LEVMARQTRAINER_H_
#define FLEX_BaseNeuralNet_LEVMARQTRAINER_H_

#include "BaseNeuralNetTrainer.h"
#include "mkl.h"
//#include "mkl_lapacke.h"

namespace flex_BaseNeuralNet
{

   template<class EFunc>
   class LevMarqTrainer : public BaseTrainer
   {
   public:
      LevMarqTrainer ();
      virtual ~LevMarqTrainer ();

      double get_mu ();

      void set_mu (double _mu);
      void set_mu_adjust (double _muInc, double _muDec);
      void set_mu_max (double _muMax);

      void init_train (BaseNeuralNet &net, const vector <Exemplar> &trainset);
      void init_training_epoch ();
      void train (BaseNeuralNet &net, const DataSet <Exemplar> &trainset);
      double sim (BaseNeuralNet &net, const DataSet <Exemplar> &trainset);
      double train_exemplar (const Exemplar &exemplar);
      void update_network_weights ();
      void update_learning_rate ();

   private:
      void update_jacobian (const BaseLayer &layer, unsigned int timeStep, unsigned int eNdx);
      void malloc_jacobian (unsigned int rows, unsigned int cols);
      void dealloc_jacobian ();
      void calc_parameter_adj ();
      void matrix_inverse (double *A, int N);
      void restore_saved_weights ();

   private:
      /* ***********************************************
       *    Stopping criteria parameters
       *
      long max_training_epochs;
      double perf_goal;
      double min_perf_gradient;
      int max_valid_fail;
      */

      /* ***********************************************
       *    Learning rate parameters
       */

      double mu;
      double mu_inc;
      double mu_dec;
      double mu_max;

      /* ************************************************
       *    Training data
       */
      EFunc output_error_functor;

//   BaseNeuralNet* neural_net;

//   double global_performance;

      vector<double> e;
      unsigned int e_ndx;
      unsigned int jacob_col_ndx;

      Array<double> jacobian;

      double *Jt_J_data; // = new double[jacobian.colDim() * jacobian.colDim()];
      double **Jt_J; // = new double*[jacobian.colDim()];

      double *JtJ_I_Jt_data; // = new double[jacobian.colDim() * jacobian.rowDim()];
      double **JtJ_I_Jt; // = new double*[jacobian.colDim()];

      double *delta; // = new double[jacobian.colDim()];

      map <string, vector<double>> delta_biases_map;
      map <string, Array<double>> delta_weights_map;

      map <string, vector<double>> saved_biases_map;
      map <string, Array<double>> saved_weights_map;
   };

   template<class EFunc>
   LevMarqTrainer<EFunc>::LevMarqTrainer () : BaseNeuralNetTrainer ()
   {
      mu = 0.001;
      mu_inc = 10;
      mu_dec = 0.1;
      mu_max = 1.0e10;
   }

   template<class EFunc>
   LevMarqTrainer<EFunc>::~LevMarqTrainer ()
   {
      Jt_J_data = 0;
      JtJ_I_Jt_data = 0;
      Jt_J = 0;
      JtJ_I_Jt = 0;
      delta = 0;
   }

/*
 * Instantiate and initialize any data structures needed to train the network as required.
 * For example structures to hold and accumulate the weight and bias deltas.
 */
   template<class EFunc>
   void LevMarqTrainer<EFunc>::init_train (BaseNeuralNet &net, const DataSet <Exemplar> &trainset)
   {
      neural_net = &net;

      const vector<BaseLayer *> network_layers = neural_net->get_network_layers ();

      /*
       * !!!!! OOOPS, the number or rows has to be the number of patterns * the number of output neurons 16 Mar 12:32 AM
       */
      unsigned int jacob_rows = trainset.size () * neural_net->get_output_size ();
      unsigned int jacob_cols = 0;

      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];

         string name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         jacob_cols += output_sz * input_sz + output_sz;

         delta_biases_map[name] = vector<double> (output_sz);
         delta_weights_map[name] = Array<double> (output_sz, input_sz);

         saved_biases_map[name] = vector<double> (output_sz);
         saved_weights_map[name] = Array<double> (output_sz, input_sz);
      }

      malloc_jacobian (jacob_rows, jacob_cols);
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::restore_saved_weights ()
   {
      const vector<BaseLayer *> network_layers = neural_net->get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];

         string name = layer.name ();

         layer.set_weights (saved_weights_map[name]);
         layer.set_biases (saved_biases_map[name]);
      }
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::malloc_jacobian (unsigned int rows, unsigned int cols)
   {
      e.resize (rows);
      jacobian.resize (rows, cols);

      Jt_J_data = new double[jacobian.colDim () * jacobian.colDim ()];

      Jt_J = new double *[jacobian.colDim ()];
      unsigned int next_col0_ndx = 0;
      for (unsigned int row = 0; row < jacobian.colDim (); row++, next_col0_ndx += jacobian.colDim ())
         Jt_J[row] = &Jt_J_data[next_col0_ndx];

      JtJ_I_Jt_data = new double[jacobian.colDim () * jacobian.rowDim ()];

      JtJ_I_Jt = new double *[jacobian.colDim ()];
      next_col0_ndx = 0;
      for (unsigned int row = 0; row < jacobian.colDim (); row++, next_col0_ndx += jacobian.rowDim ())
         JtJ_I_Jt[row] = &JtJ_I_Jt_data[next_col0_ndx];

      delta = new double[jacobian.colDim ()];

   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::dealloc_jacobian ()
   {
      delete[] Jt_J_data;
      Jt_J_data = 0;

      delete[] JtJ_I_Jt_data;
      JtJ_I_Jt_data = 0;

      delete[] Jt_J;
      Jt_J = 0;

      delete[] JtJ_I_Jt;
      JtJ_I_Jt = 0;

      delete[] delta;
      delta = 0;

      jacobian.resize (0, 0);
      e.resize (0);
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::train (BaseNeuralNet &net, const DataSet <Exemplar> &trainset)
   {
      bool done = false;
      long epoch_no = 0;
      long retries = 0;
      unsigned int trainset_ndx;
      double perf;

      neural_net = &net;

      init_train (net, trainset);

      double adj_global_perf;
      global_performance = 0;
      while (!done)
      {
         cout << ">>> epoch no " << epoch_no << "<<<" << endl;
         global_performance = 0;
         init_training_epoch ();

         for (trainset_ndx = 0; trainset_ndx < trainset.size (); trainset_ndx++)
         {
            perf = train_exemplar (trainset.at (trainset_ndx));
            global_performance += perf;
         }
         global_performance /= trainset.size ();
         cout << ">>>>> global perf before update= " << global_performance << endl;

         update_network_weights ();
         adj_global_perf = sim (*neural_net, trainset);

         cout << ">>>>> adj global perf = " << adj_global_perf << endl;

         /*
          * Adjustment decreased the error so continue to next epoch
          */
         if (adj_global_perf < global_performance)
         {
            mu = mu * mu_dec;
            epoch_no++;
            retries = 0;
            if (epoch_no > max_training_epochs)
               done = true;
         }
            /*
             * Else; adjustment increased error so restore saved weights
             * adjust mu and try again
             */
         else
         {
            restore_saved_weights ();

            if (mu == mu_max)
            {
               cout << "cant converge at mu_max" << endl;
               done = true;
            }

            mu = mu * mu_inc;
            if (mu > mu_max)
               mu = mu_max;

            cout << ">>>>> global perf increased, " << global_performance << " to " << adj_global_perf << endl;

            /*
            if (retries < 5)
                cout << ">>>>> restore weights, adjust mu and try again <<<" << endl;
            else
            {
               cout << ">>>>> to many retries(" << retries << "), done <<<<" << endl;
               done = true;
            }
            */

            retries++;
         }

         //update_learning_rate();
         //update_performance_trace();



         //done = is_training_complete(epoch_no);
      }
   }

   template<class EFunc>
   double LevMarqTrainer<EFunc>::sim (BaseNeuralNet &net, const DataSet <Exemplar> &trainset)
   {
      long epoch_no;
      unsigned int trainset_ndx;
      double perf;

      neural_net = &net;

      double my_global_performance = 0;

      for (trainset_ndx = 0; trainset_ndx < trainset.size (); trainset_ndx++)
      {
         const Exemplar &exemplar = trainset.at (trainset_ndx);

         vector<double> inpattern = exemplar.input ();
         vector<double> tgtvec = exemplar.target_output ();

         // Present input pattern and get output
         const vector<double> &netout = (*this->neural_net) (exemplar.input ());

         // Calculate the output error
         double isse;
         vector<double> gradient (netout.size ());
         output_error_functor (isse, gradient, netout, tgtvec);

         my_global_performance += isse;

      }
      my_global_performance /= trainset.size ();

      return my_global_performance;
   }

/*
 * Perform any initialization required for the new training epoch. For example clear
 * all data structures required to accumulate the new global network error, the weight
 * and bias deltas and etc.
 */
   template<class EFunc>
   void LevMarqTrainer<EFunc>::init_training_epoch ()
   {
      neural_net->clear_error ();
      e_ndx = 0;
      jacob_col_ndx = 0;

      /* ******************************************
       *    Zero bias and weight deltas
       */
      const vector<BaseLayer *> network_layers = neural_net->get_network_layers ();

      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         string name = layer.name ();

         vector<double> &saved_bias = saved_biases_map[name];
         Array<double> &saved_weights = saved_weights_map[name];

         for (unsigned int i = 0; i < saved_bias.size (); i++)
            saved_bias[i] = layer.get_biases ().at (i);
         saved_weights = layer.get_weights ();
      }
   }

   template<class EFunc>
   double LevMarqTrainer<EFunc>::train_exemplar (const Exemplar &exemplar)
   {
      vector<double> inpattern = exemplar.input ();
      vector<double> tgtvec = exemplar.target_output ();

      // Present input pattern and get output
      const vector<double> &netout = (*this->neural_net) (exemplar.input ());

      // Calculate the output error
      double isse;
      vector<double> gradient (netout.size ());
      output_error_functor (isse, gradient, netout, tgtvec);

      /*
       * LevMarq special - add the gradient values for the output vector for this pattern to "e"
       *
      for (unsigned int i=0; i<gradient.size(); i++)
         e.at(e_ndx++) = gradient.at(i);
      */

      cout << "isse " << isse << endl;

      // Backprop the error through the network

      /*
       * For Levenberg-Marquet we only backprop the gradient of the output
       * transfer function at this point
       */
      vector<double> unit_gradient (netout.size ());
      for (unsigned int i = 0; i < gradient.size (); i++)
      {
         for (unsigned int j = 0; j < gradient.size (); j++)
         {
            unit_gradient[j] = 0.0;
            if (i == j)
               unit_gradient[j] = -1.0;
         }

         this->neural_net->clear_error ();
         this->neural_net->backprop (unit_gradient);

         jacob_col_ndx = 0;
         const vector<BaseLayer *> network_layers = this->neural_net->get_network_layers ();
         for (unsigned int layer_ndx = 0; layer_ndx < network_layers.size (); layer_ndx++)
         {
            BaseLayer &layer = *network_layers[layer_ndx];
            update_jacobian (layer, 1, e_ndx);
         }

//      cout << "e[" << e_ndx << "] = " << e.at(e_ndx) << endl;
         e.at (e_ndx++) = gradient.at (i);
      }

      return isse;
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::update_jacobian (const BaseLayer &layer, unsigned int timeStep, unsigned int eNdx)
   {
      const vector<double> &errorv = layer.get_error (timeStep);
      const Array<double> &dNdW = layer.get_dNdW (timeStep);
      const vector<double> &dAdN = layer.get_dAdN (timeStep);

      unsigned int layer_input_size = layer.input_size ();
      for (unsigned int out_ndx = 0; out_ndx < layer.size (); out_ndx++)
      {
         for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
         {
            // errorv here is just the backpropogated gradient for levmarq
            jacobian.at (eNdx, jacob_col_ndx++) =
               errorv.at (out_ndx) * dAdN.at (out_ndx) * dNdW.at (out_ndx, in_ndx);
         }
      }

      const vector<double> &dAdB = layer.get_dAdB (timeStep);

      for (unsigned int i = 0; i < layer.size (); i++)
      {
         // errorv here is just the backpropogated gradient for levmarq
         jacobian.at (eNdx, jacob_col_ndx++) = errorv.at (i) * dAdB.at (i);
      }
   }

/*
 * Update the network layer weights and biases based on the calculated weight and bias deltas.
 */
   template<class EFunc>
   void LevMarqTrainer<EFunc>::update_network_weights ()
   {
      calc_parameter_adj ();

      const vector<BaseLayer *> network_layers = neural_net->get_network_layers ();

      cout << "weight adjustments" << endl;
      unsigned int delta_ndx = 0;
      for (unsigned int layer_ndx = 0; layer_ndx < network_layers.size (); layer_ndx++)
      {
         BaseLayer &layer = *network_layers[layer_ndx];
         string name = layer.name ();

         vector<double> &delta_bias = this->delta_biases_map[name];
         Array<double> &delta_weights = this->delta_weights_map[name];

         unsigned int layer_input_size = layer.input_size ();
         for (unsigned int out_sz = 0; out_sz < layer.size (); out_sz++)
         {
            for (unsigned int in_sz = 0; in_sz < layer_input_size; in_sz++)
            {
               cout << delta[delta_ndx] << " ";
               delta_weights.at (out_sz, in_sz) = -delta[delta_ndx++];
            }
         }
         layer.adjust_weights (delta_weights);

         for (unsigned int i = 0; i < layer.size (); i++)
         {
            cout << delta[delta_ndx] << " ";
            delta_bias.at (i) = -delta[delta_ndx++];
         }
         layer.adjust_biases (delta_bias);

      }
      cout << endl;
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::calc_parameter_adj ()
   {
// Calculate J'*J
      for (unsigned int jj_row = 0; jj_row < jacobian.colDim (); jj_row++)
      {
         for (unsigned int jj_col = 0; jj_col < jacobian.colDim (); jj_col++)
         {
            Jt_J[jj_row][jj_col] = 0;
            for (unsigned int jt_col = 0; jt_col < jacobian.rowDim (); jt_col++)
               Jt_J[jj_row][jj_col] += jacobian.at (jt_col, jj_col) * jacobian.at (jt_col, jj_col);
         }
      }

      // Add mu*I
      for (unsigned int jj_row = 0; jj_row < jacobian.colDim (); jj_row++)
         Jt_J[jj_row][jj_row] += mu;

      matrix_inverse (Jt_J_data, jacobian.colDim ());

      // Calculate (J'*J + muI)_inv * J'
      for (unsigned int jj_row = 0; jj_row < jacobian.colDim (); jj_row++)
      {
         for (unsigned int jj_col = 0; jj_col < jacobian.rowDim (); jj_col++)
         {
            JtJ_I_Jt[jj_row][jj_col] = 0;
            for (unsigned int jt_col = 0; jt_col < jacobian.colDim (); jt_col++)
               JtJ_I_Jt[jj_row][jj_col] += Jt_J[jj_row][jt_col] * jacobian.at (jj_col,
                                                                               jt_col);
         }
      }

      // Multiply by e to get deltas
      for (unsigned int jj_row = 0; jj_row < jacobian.colDim (); jj_row++)
      {
         delta[jj_row] = 0;
         for (unsigned int jt_col = 0; jt_col < jacobian.rowDim (); jt_col++)
            delta[jj_row] += JtJ_I_Jt[jj_row][jt_col] * e.at (jt_col);
      }
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::matrix_inverse (double *A, int N)
   {
      int *IPIV = new int[N + 1];
      int LWORK = N * N;
      double *WORK = new double[LWORK];
      int INFO;

      /*
      dgetrf(&N,&N,A,&N,IPIV,&INFO);
      dgetri(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
      */

      INFO = LAPACKE_dgetrf (LAPACK_ROW_MAJOR, N, N, A, N, IPIV);
      cout << "lu decomposition info " << INFO << endl;

      INFO = LAPACKE_dgetri (LAPACK_ROW_MAJOR, N, A, N, IPIV);
      cout << "inverse info " << INFO << endl;

      delete IPIV;
//    delete WORK;
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::update_learning_rate ()
   {
      // NOP for Gradient Descent algorithm
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::set_mu (double _mu)
   {
      this->mu = _mu;
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::set_mu_adjust (double _muInc, double _muDec)
   {
      mu_inc = _muInc;
      mu_dec = _muDec;
   }

   template<class EFunc>
   void LevMarqTrainer<EFunc>::set_mu_max (double _muMax)
   {
      mu_max = _muMax;
   }

   template<class EFunc>
   double LevMarqTrainer<EFunc>::get_mu ()
   {
      return mu;
   }

} /* namespace flex_BaseNeuralNet */

#endif /* FLEX_BaseNeuralNet_GDTRAINER_H_ */
