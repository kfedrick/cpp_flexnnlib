//
// Created by kfedrick on 3/3/21.
//

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>

#include "picojson.h"

#include "flexnnet.h"
#include "CartesianCoord.h"
#include "EnumeratedDataSet.h"
#include "EnumeratedDataSetStream.h"

using flexnnet::CartesianCoord;
using flexnnet::EnumeratedDataSet;

class DataSetTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void
   SetUp()
   {}
   virtual void
   TearDown()
   {}
};

TEST_F(DataSetTestFixture, CCoordConstructor)
{
   std::cout << "Test CartesianCoord Constructor\n" << std::flush;

   CartesianCoord coord(5,5);
}

TEST_F(DataSetTestFixture, CCoordValueMap)
{
   std::cout << "Test CartesianCoord value_map\n" << std::flush;

   CartesianCoord coord(0,0);
   flexnnet::ValarrMap m = coord.value_map();

   std::cout << this->prettyPrintVector("x", m["x"]).c_str() << "\n";
   std::cout << this->prettyPrintVector("y", m["y"]).c_str() << "\n";
}

TEST_F(DataSetTestFixture, DatasetConstructor)
{
   std::cout << "Test Empty Dataset Constructor\n" << std::flush;

   EnumeratedDataSet<CartesianCoord, std::valarray<double>> dataset;

   ASSERT_EQ(dataset.size(), 0);
}

TEST_F(DataSetTestFixture, DatasetPushback)
{
   std::cout << "Test Empty Dataset Pushback\n" << std::flush;

   EnumeratedDataSet<CartesianCoord, std::valarray<double>> dataset;

   dataset.push_back(std::pair<CartesianCoord, std::valarray<double>>(CartesianCoord(0,0), {1}));
   ASSERT_EQ(dataset.size(), 1);
}




TEST_F(DataSetTestFixture, Picojson)
{
   std::cout << "Test Dataset Picojson\n" << std::flush;

   std::string objjson = "{\"x\":3.14159,\"y\":2.78}";
   std::stringstream ss;
   picojson::value v;

   ss.str(objjson);
   ss >> v;

   if (v.is<picojson::object>())
   {
      std::cout << "is object\n" << std::flush;
      const picojson::object& o = v.get<picojson::object>();
      for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i)
         std::cout << i->first << "  " << i->second << std::endl;
   }
   else
      std::cout << "not object\n" << std::flush;

   std::string arrjson = "[3.14159, 2.78]";
   ss.str(arrjson);
   ss >> v;

   if (v.is<picojson::array>())
   {
      std::cout << "is array\n" << std::flush;
      const picojson::array& a = v.get<picojson::array>();
      for (picojson::array::const_iterator i = a.begin(); i != a.end(); ++i)
         std::cout << "  " << *i << std::endl;
   }
   else
      std::cout << "not array\n" << std::flush;
}

TEST_F(DataSetTestFixture, PicojsonPartial)
{
   std::cout << "Test Dataset PicojsonPartial\n" << std::flush;

   std::string objjson = "{\"x\":3.14159,\"y\":2.78},{\"x\":0.6,\"y\":5.9}";
   std::stringstream ss;
   picojson::value v, v2, v3;

   std::cout << "Get first\n" << std::flush;
   ss.str(objjson);
   ss >> v;

   if (v.is<picojson::object>())
   {
      std::cout << "is object\n" << std::flush;
      const picojson::object& o = v.get<picojson::object>();
      for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i)
         std::cout << i->first << "  " << i->second << std::endl;
   }
   else
      std::cout << "not object\n" << std::flush;

   std::cout << "Get next\n" << (char)ss.get() << std::flush;
   ss >> v2;
   if (v2.is<picojson::object>())
   {
      std::cout << "is object\n" << std::flush;
      const picojson::object& o = v2.get<picojson::object>();
      for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i)
         std::cout << i->first << "  " << i->second << std::endl;
   }
   else
      std::cout << "not object\n" << std::flush;
}

TEST_F(DataSetTestFixture, PicojsonCoord)
{
   std::cout << "Test Dataset PicoReadCoord\n" << std::flush;

   std::string objjson = "{\"x\":3.14159,\"y\":2.78}";
   std::stringstream ss;
   CartesianCoord coord;

   ss.str(objjson);
   ss >> coord;

   std::cout << coord << "\n" << std::flush;
}

TEST_F(DataSetTestFixture, PicojsonCoordList)
{
   std::cout << "Test Dataset PicoReadCoordList\n" << std::flush;

   std::string objjson = "{\"x\":3.14159,\"y\":2.78},{\"x\":0.6,\"y\":5.9}";
   std::stringstream ss;
   CartesianCoord coord1, coord2;

   ss.str(objjson);
   ss >> coord1;

   // skip past comma
   (char)ss.get();

   ss >> coord2;

   std::cout << coord1 << "\n" << std::flush;
   std::cout << coord2 << "\n" << std::flush;

   std::cout << "Done!\n" << std::flush;
}

TEST_F(DataSetTestFixture, OStream)
{
   std::cout << "Test Dataset Console Stream\n" << std::flush;

   EnumeratedDataSet<CartesianCoord, CartesianCoord> dataset;

   dataset.push_back(std::pair<CartesianCoord, CartesianCoord>(CartesianCoord(0,0), CartesianCoord(0,0)));
   dataset.push_back(std::pair<CartesianCoord, CartesianCoord>(CartesianCoord(1,3), CartesianCoord(0,0)));
   dataset.push_back(std::pair<CartesianCoord, CartesianCoord>(CartesianCoord(-1,0.3), CartesianCoord(0,0)));

   std::cout << dataset;
}

TEST_F(DataSetTestFixture, Write)
{
   std::cout << "Test Dataset File Stream Write\n" << std::flush;

   std::string fname = "test_write.txt";

   EnumeratedDataSet<CartesianCoord, CartesianCoord> dataset;

   dataset.push_back(std::pair<CartesianCoord, CartesianCoord>(CartesianCoord(0,0), CartesianCoord(0,0)));
   dataset.push_back(std::pair<CartesianCoord, CartesianCoord>(CartesianCoord(1,3), CartesianCoord(0,0)));
   dataset.push_back(std::pair<CartesianCoord, CartesianCoord>(CartesianCoord(-1,0.3), CartesianCoord(0,13)));

   std::ofstream of_strm(fname);
   of_strm << dataset;
   of_strm.close();
}

TEST_F(DataSetTestFixture, IStream)
{
   std::cout << "Test Dataset Input File Stream\n" << std::flush;

   std::string fname = "test_write.txt";

   EnumeratedDataSet<CartesianCoord, CartesianCoord> dataset;

   // Open file for writing
   std::ifstream if_strm(fname);
   if_strm >> dataset;

   ASSERT_EQ(dataset.size() , 3) << "Unexpected data set size" << dataset.size();


   std::cout << "Read exemplar:\n" << std::flush;
   for (auto& exemplar : dataset)
   {
      std::cout << exemplar.first;
      std::cout << exemplar.second;
   }
   std::cout << "\n-----------------------\n" << std::flush;

   if_strm.close();
}



int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}