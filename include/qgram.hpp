/* DOCUMENTATION

*/



#ifndef _QGRAM_H_
#define _QGRAM_H_



#include <vector>
#include <string>
#include <unordered_map>



using namespace std;



/* DOCUMENTATION
 *
 */
typedef struct qgram_t {

} qgram_t;



/* DOCUMENTATION
 *
 */
vector<string> qg_get_record (string str, int qgramsize);



/* DOCUMENTATION
 *
 */
vector< vector<string> > qg_get_records (vector<string> data, int qgramsize);



/* DOCUMENTATION
 *
 */
vector< vector<unsigned long> >
qg_get_sets (vector< vector<string> > recs);



/* DOCUMENTATION
 *
 */
void qg_print_record  (vector<string> rec);
void qg_print_records (vector< vector<string> > recs);
void qg_print_set     (vector<unsigned long> set);
void qg_print_sets    (vector< vector<unsigned long> > sets);



/* DOCUMENTATION
 *
 */
unsigned long qg_hash (string qgram);



#endif // _QGRAM_H_
