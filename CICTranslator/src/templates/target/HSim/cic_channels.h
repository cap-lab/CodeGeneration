typedef enum {
    CHANNEL_TYPE_NORMAL,
    CHANNEL_TYPE_ARRAY_CHANNEL,
} CHANNEL_TYPE;

typedef struct {
    int channel_id;
    CHANNEL_TYPE type;
    volatile unsigned int *buf;
    int max_size;
    int initData;
    int sampleSize;
    volatile int *waiting_reader_task_list;
    int waiting_reader_task_list_size;
    volatile int *waiting_writer_task_list;
    int waiting_writer_task_list_size;
} CHANNEL;