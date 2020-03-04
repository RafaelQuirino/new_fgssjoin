/* DOCUMENTATION

*/



#ifndef _IO_H_
#define _IO_H_



#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "util.hpp"



using namespace std;



/* DOCUMENTATION
 *
 */
vector<string> io_getlines (string filename);



/* DOCUMENTATION
 *
 */
vector<string> io_getnlines (string filename, int nlines);



#endif // _IO_H_

