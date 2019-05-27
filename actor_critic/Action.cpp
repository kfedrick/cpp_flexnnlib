/*
 * Action.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include "Action.h"

namespace flexnnet
{

   Action::Action (const char *_name) :
      NamedObject (_name)
   {
      id = NO_ACTION;
   }

   Action::Action (const string &_name) :
      NamedObject (_name)
   {
      id = NO_ACTION;
   }

   Action::~Action ()
   {
   }

   Action::Action (const string &_name, unsigned int _id) :
      NamedObject (_name)
   {
      id = _id;
   }

   Action::Action (const string &_name, unsigned int _id,
                   const vector<double> &_parameters) :
      NamedObject (_name)
   {
      id = _id;
      parameters = _parameters;
   }

   Action::Action (const Action &_action) :
      NamedObject (_action.name ())
   {
      copy (_action);
   }

   unsigned int Action::action_id () const
   {
      return id;
   }

   void Action::action_id (unsigned int _id)
   {
      id = _id;
   }

   void Action::action_parameters (const vector<double> &_parameters)
   {
      parameters = _parameters;
   }

   const vector<double> &Action::action_parameters () const
   {
      return parameters;
   }

   void Action::clear_action_parameters ()
   {
      parameters.clear ();
   }

   Action &Action::operator= (const Action &_action)
   {
      copy (_action);
      return *this;
   }

   void Action::copy (const Action &_action)
   {
      NamedObject::copy (_action);
      id = _action.id;
      parameters = _action.parameters;
   }

} /* namespace flexnnet */


