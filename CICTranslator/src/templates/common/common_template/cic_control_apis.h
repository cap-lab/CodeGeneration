#ifndef __CIC_CONTROL_APIS_H__
#define __CIC_CONTROL_APIS_H__

CIC_EXTERN CIC_T_LONG 		GetParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name);
CIC_EXTERN CIC_T_VOID 		SetParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_LONG p_value);
CIC_EXTERN CIC_T_DOUBLE 	GetParamFloat(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name);
CIC_EXTERN CIC_T_VOID 		SetParamFloat(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_DOUBLE p_value);
CIC_EXTERN CIC_T_VOID 		RunTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		StopTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		SuspendTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_namet);                             
CIC_EXTERN CIC_T_VOID 		ResumeTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		CallTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_INT 		GetCurrentTimeBase(CIC_T_VOID);
CIC_EXTERN CIC_T_INT 		SetTimer(CIC_T_CHAR* caller_task_name, CIC_T_UINT time_value, CIC_T_CHAR *time_unit);
CIC_EXTERN CIC_T_INT 		GetTimerAlarmed(CIC_T_CHAR* caller_task_name, CIC_T_UINT timer_id);
CIC_EXTERN CIC_T_VOID 		ResetTimer(CIC_T_CHAR* caller_task_name, CIC_T_UINT timer_id);
CIC_EXTERN CIC_T_VOID 		ProgrammKill(CIC_T_VOID);
CIC_EXTERN CIC_T_VOID 		ProgramStop(CIC_T_CHAR* caller_task_name);

#define SYS_REQ_GET_PARAM_INT(a, b)				 GetParamInt(TASK_NAME, a, b)
#define SYS_REQ_SET_PARAM_INT(a, b, c)	 		 SetParamInt(TASK_NAME, a, b, c)
#define SYS_REQ_GET_PARAM_FLOAT(a, b)			 GetParamFloat(TASK_NAME, a, b)
#define SYS_REQ_SET_PARAM_FLOAT(a, b, c)		 SetParamFloat(TASK_NAME, a, b, c)
#define SYS_REQ_RUN_TASK(a)						 RunTask(TASK_NAME, a)
#define SYS_REQ_STOP_TASK(a)					 StopTask(TASK_NAME, a)
#define SYS_REQ_SUSPEND_TASK(a)					 SuspendTask(TASK_NAME, a)                                                     
#define SYS_REQ_RESUME_TASK(a)					 ResumeTask(TASK_NAME, a)
#define SYS_REQ_CALL_TASK(a)					 CallTask(TASK_NAME, a)
#define SYS_REQ_GET_CURRENT_TIME_BASE()			 GetCurrentTimeBase()
#define SYS_REQ_SET_TIMER(a, b)					 SetTimer(TASK_NAME, a, b)
#define SYS_REQ_GET_TIMER_ALARMED(a)			 GetTimerAlarmed(TASK_NAME, a)
#define SYS_REQ_RESET_TIMER(a) 					 ResetTimer(TASK_NAME, a)
#define SYS_REQ_KILL() 							 ProgramKill()
#define SYS_REQ_STOP() 							 ProgramStop(TASK_NAME)

#endif /* __CIC_CONTROL_APIS_H__ */

