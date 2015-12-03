
##HEADER_INCLUDE

typedef struct {
    unsigned int channel_id;
    char *lib_name;
    char op;
 
    BUFFER_DATA_TYPE buf;
    BUFFER_DATA_TYPE start;
    BUFFER_DATA_TYPE end;
    int max_size;
    int cur_size;
    int sampleSize;
    int isFull;

    MUTEX_TYPE mutex;
    COND_TYPE cond;
} LIB_CHANNEL;
