/*
 * NamedObject.h
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NAMEDOBJECT_H_
#define FLEX_NEURALNET_NAMEDOBJECT_H_

#include <string>

namespace flexnnet
{
   class NamedObject
   {
   public:

      NamedObject(const char* _name = "NamedObject");
      NamedObject(const std::string& _name);

      void rename(const std::string& _name);
      void rename(const std::string&& _name);

      const std::string& name() const;

   protected:
      void copy(const NamedObject& _namedObj);

   private:
      std::string myname;
   };

   inline NamedObject::NamedObject(const std::string& _name)
   {
      myname = _name;
   }

   inline NamedObject::NamedObject(const char* _name)
   {
      myname = _name;
   }

   inline const std::string& NamedObject::name() const
   {
      return myname;
   }

   inline void NamedObject::copy(const NamedObject& _namedObj)
   {
      myname = _namedObj.myname;
   }

   inline void NamedObject::rename(const std::string& _name)
   {
      myname = _name;
   }

   inline void NamedObject::rename(const std::string&& _name)
   {
      myname = std::move(_name);
   }
} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_NAMEDOBJECT_H_ */
