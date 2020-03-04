#include "io.hpp"



/* DOCUMENTATION
 *
 */
vector<string> io_getlines (string filename)
{
    ifstream input(filename.c_str());
    string line;

    vector<string> lines;

    if (input.is_open()) {
	    while (!input.eof()) {
	        getline(input, line);
	        if(line == "") continue;
			lines.push_back(line);
	    }
    }

    input.close();

    return lines;
}



/* DOCUMENTATION
 *
 */
vector<string> io_getnlines (string filename, int n)
{
    ifstream input(filename.c_str());
    string line;

    vector<string> lines;
    int i = 0;
    if (input.is_open()) {
        while (!input.eof() && i++ < n) {
            getline(input, line);
            if(line == "") continue;
            lines.push_back(line);

            if (i >= n) break;
        }
    }

    input.close();

    return lines;
}
