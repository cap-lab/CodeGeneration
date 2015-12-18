#ifndef __CIC_MTM_APIS_H__
#define __CIC_MTM_APIS_H__

CIC_EXTERN CIC_T_VOID 		ExecuteTransition(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		SetMTMParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_LONG p_value);
CIC_EXTERN CIC_T_VOID 		SetMTMParamString(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_CHAR* p_value);
CIC_EXTERN CIC_T_CHAR* 		GetMode(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_INT 		GetCurrentModeName(CIC_T_CHAR* caller_task_name); 

#define SYS_REQ_EXECUTE_TRANSITION(a) 			 ExecuteTransition(TASK_NAME, a)
#define SYS_REQ_SET_MTM_PARAM_INT(a, b, c) 		 SetMTMParamInt(TASK_NAME, a, b, c)
#define SYS_REQ_SET_MTM_PARAM_STRING(a, b, c) 	 SetMTMParamString(TASK_NAME, a, b, c)
#define SYS_REQ_GET_MODE(a) 					 GetMode(TASK_NAME, a)
#define SYS_REQ_GET_CURRENT_MODE_NAME(a) 		 GetCurrentModeName(TASK_NAME)  

#endif /* __CIC_MTM_APIS_H__ */

