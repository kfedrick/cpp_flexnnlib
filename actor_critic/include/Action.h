/*
 * Action.h
 *
 *  Created on: Mar 17, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ACTION_H_
#define FLEX_NEURALNET_ACTION_H_

#include <vector>
#include <string>

#include "NamedObject.h"

using namespace std;

namespace flex_neuralnet
{

class Action: public NamedObject
{
public:
   static const unsigned int NO_ACTION = 0;

public:
   Action(const char* _name = "NO_ACTION");
   Action(const string& _name);
   Action(const string& _name, unsigned int _id);
   Action(const string& _name, unsigned int _id,
         const vector<double>& _params);
   Action(const Action& _action);
   ~Action();

   unsigned int action_id() const;
   void action_id(unsigned int _id);
   void action_parameters(const vector<double>& _parameters);
   const vector<double>& action_parameters() const;
   void clear_action_parameters();

   Action& operator=(const Action& _action);

private:
   void copy(const Action& _action);

private:
   unsigned int id;
   vector<double> parameters;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ACTION_H_ */
