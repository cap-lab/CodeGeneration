typedef struct {
        int task_id;
        int cur_size;
        int wait_size;
        int max_size;
        int sample_size;
        char direction;
        unsigned char *buf;
        unsigned char *start;
        unsigned char *end;
        MUTEX_TYPE p_mutex;
} CLUSTER_BUFFER;

typedef struct {                                                                                                                       
        int task_id;                                                                                                                   
        int isGPU;
        int gpu_index;                                                                                                                    
} GPUTASKINFO;  

typedef struct {
        int channel_id;
        unsigned char* read_wait;
        unsigned char* write_wait;
        int wait_size;
        int cpu_gpu;
} GPUCHANNELINFO;
