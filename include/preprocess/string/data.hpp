/* DOCUMENTATION

*/

#ifndef _DATA_H_
#define _DATA_H_

#include <vector>
#include <string>



using namespace std;



/* DOCUMENTATION
 *
 */
vector<string>
dat_get_input_data (string file_path, int input_size);



/* DOCUMENTATION
 *
 */
vector<string>
dat_get_proc_data (vector<string>& input_data);



/* DOCUMENTATION
 *
 */
void
dat_proc_data (vector<string>& input_data);



#endif // _DATA_H_
