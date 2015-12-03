#include <stdio.h>
#include <sys/time.h>

#define TIMESTAMP(a, b, c, d, e, f) time_check(a, b, c, d, e, f)

typedef struct {
    int start_end;
    int task_library;
    int task_id;
    int library_id;
    int func_id;
    long sec_time_record;
    long usec_time_record;
}save_unit;

static save_unit save;

void time_check(FILE* trace, int start_end, int task_library, int task_id, int library_id, int func_id)
{
	struct timeval check_time;
	long sec_time_record=0;
	long usec_time_record=0;

	gettimeofday(&check_time, 0); 
	sec_time_record = check_time.tv_sec % 100;
	usec_time_record = check_time.tv_usec;

    save.start_end = start_end;
    save.task_library = task_library;
    save.task_id = task_id;
    save.library_id = library_id;
    save.func_id = func_id;
    save.sec_time_record = sec_time_record;
    save.usec_time_record = usec_time_record;

    fwrite(&save, sizeof(save_unit), 1, trace);
}

    // 파일 다시 열어서 index들을 string으로 바꿔주기
    /*
    FILE* trace_1;
    FILE* trace_2;

    trace_1 = fopen("trace_library", "r");
    trace_2 = fopen("trace_library_temp", "w");
    while(&save != NULL)
    {
	    count++;
        fread(&save, sizeof(save_unit), 1, trace_1);
	    fprintf(trace_2, "count : %.3d, start_end : %.3d, task_library : %.3d, task_id : %.3d, library_id : %.3d, func_id : %.3d, time : %ldsec %ldusec\n", count, save.start_end, save.task_library, save.task_id, save.library_id, save.func_id, save.sec_time_record, save.usec_time_record);
    }
    fclose(trace_1);
    fclose(trace_2);
    */


