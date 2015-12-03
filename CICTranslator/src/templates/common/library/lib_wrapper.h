typedef struct {
    int task_id;
    int proc_id;
    char *name;
    voidfunc init;
    voidfunc go;
    voidfunc wrapup;
    THREAD_TYPE th;
} LIB_WRAPPER;


