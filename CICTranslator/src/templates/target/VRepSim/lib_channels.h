
##HEADER_INCLUDE

typedef struct {
    unsigned int channel_id;
    char lib_name[32];
    char op;

    unsigned char buf[1024];
    int start;
    int end;
    int max_size;
    int cur_size;
    int sampleSize;
    int isFull;

    MUTEX_TYPE mutex;
    COND_TYPE cond;

    int canIStart;
} LIB_CHANNEL;
