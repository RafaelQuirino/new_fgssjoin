/*    
    // CONSTANTS
    string file_path(argv[1]);
    int input_size = 0;
    int qgram_size = atoi(argv[2]);


    // VARIABLES
    unsigned long                        n_sets;
    unsigned long                        n_terms;
    unsigned long                        n_tokens;
    vector<string>                       raw_data;
    vector<string>                       proc_data;
    vector<size_t>                       doc_index;
    vector< vector<string> >             docs;
    unordered_map<unsigned long,token_t> dict;
    vector< vector<token_t> >            tsets;


    // MEASURING PERFORMANCE (in multiple leves)
    unsigned long t0, t1, t00, t01, t000, t001;
    double total_time;


    t0 = ut_get_time_in_microseconds();


    // READING DATA AND CREATING RECORDS ---------------------------------------
    fprintf(stderr, "Reading data and creating records...\n");
    t00 = ut_get_time_in_microseconds();

    // READING INPUT DATA
    raw_data = dat_get_input_data(file_path, input_size);

    printf("\traw_data size: %zu\n", raw_data.size());

    // PROCESSING DATA
    proc_data = dat_get_proc_data(raw_data);

    // CREATING QGRAM RECORDS
    docs = qg_get_records(proc_data, qgram_size);

    t01 = ut_get_time_in_microseconds();
    fprintf(stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds(t00, t01));
    //--------------------------------------------------------------------------

    // qg_print_records(docs);


    // CREATING TOKEN DICTIONARY -----------------------------------------------
    fprintf(stderr, "Creating token dictionary...\n");
    t00 = ut_get_time_in_microseconds();

    dict = tk_get_dict(docs);

    n_sets   = docs.size();
    n_terms  = dict.size();
    n_tokens = 0;
    for (int i = 0; i < docs.size(); i++)
    {
        n_tokens += docs[i].size();
    }

    t01 = ut_get_time_in_microseconds();
    fprintf(stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds(t00, t01));

    fprintf(stderr, "n_sets  : %lu\n", n_sets);
    fprintf(stderr, "n_terms : %lu\n", n_terms);
    fprintf(stderr, "n_tokens: %lu\n\n", n_tokens);
    //--------------------------------------------------------------------------

    // tk_print_dict(dict);


    // CREATING TOKEN SETS -----------------------------------------------------
    fprintf(stderr, "Creating token sets...\n");
    t00 = ut_get_time_in_microseconds();

    tsets = tk_get_tokensets(dict, docs, doc_index);

    t01 = ut_get_time_in_microseconds();
    fprintf(stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds(t00, t01));
    //--------------------------------------------------------------------------

    // tk_print_tsets(tsets, ORDER_ID);


    // SORTING AND PREPARING TOKENSETS -----------------------------------------
    fprintf (stderr, "Preparing token sets...\n");
    t00 = ut_get_time_in_microseconds();

    // Sort sets in decreasing size order.
    fprintf(stderr, "\tSorting corpus by set by size...\n");
    t000 = ut_get_time_in_microseconds();

    doc_index = tk_sort_sets (tsets);

    t001 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done. It took %gms.\n",
        ut_interval_in_miliseconds (t000, t001)
    );

    // fprintf (stderr, "\tSorting each set by term frequency...\n");
    // t000 = ut_get_time_in_microseconds();

    // tk_sort_freq (tsets);

    // t001 = ut_get_time_in_microseconds();
    // fprintf(stderr, "\t> Done. It took %gms.\n",
    //     ut_interval_in_miliseconds (t000, t001)
    // );

    t01 = ut_get_time_in_microseconds();
    fprintf (stderr, "> Done in %gms.\n\n", ut_interval_in_miliseconds (t00, t01));
    //--------------------------------------------------------------------------


    t1 = ut_get_time_in_microseconds();
    fprintf(stderr, 
        "Pre-processing %s took %gms.\n\n", 
        file_path.c_str(), 
        ut_interval_in_miliseconds(t0, t1)
    );
*/