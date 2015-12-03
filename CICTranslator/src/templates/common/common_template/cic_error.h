#ifndef __CIC_ERROR_H__
#define __CIC_ERROR_H__

#define CIC_V_ERROR_EXIT 0
#define CIC_V_ERROR_CONTINUE 1
#define CIC_V_ERROR -1

#define CIC_F_ERROR ErrorHandling

CIC_STATIC CIC_T_VOID ErrorHandling(CIC_T_CHAR* task_name, CIC_T_CHAR* error_msg, CIC_T_INT flag){
    CIC_F_PRINT_STRING("ERROR] ");
	CIC_F_PRINT_STRING(task_name);
	CIC_F_PRINT_STRING(": ");
	CIC_F_PRINT_STRING(error_msg);
	CIC_F_PRINT_STRING("\n");
	
	if(flag == CIC_V_ERROR_EXIT)	CIC_F_EXIT(CIC_V_ERROR);
}

#endif /* __CIC_ERROR_H__ */