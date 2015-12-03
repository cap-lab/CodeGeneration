#ifndef __CIC_APIS_H__
#define __CIC_APIS_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

CIC_EXTERN CIC_T_INT InitPort(CIC_T_INT task_id, CIC_T_INT port_id);
CIC_EXTERN CIC_T_INT InitTaskPort(CIC_T_INT task_id, CIC_CONST CIC_T_CHAR* port_name);
CIC_EXTERN CIC_T_INT ReadPort(CIC_T_INT channel_id, CIC_T_UCHAR *buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT WritePort(CIC_T_INT channel_id, CIC_T_UCHAR *buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT Available(CIC_T_INT channel_id);
CIC_EXTERN CIC_T_INT ReadACPort(CIC_T_INT channel_id, CIC_T_UCHAR *buf, CIC_T_INT len, CIC_T_INT index);
CIC_EXTERN CIC_T_INT WriteACPort(CIC_T_INT channel_id, CIC_T_UCHAR *buf, CIC_T_INT len, CIC_T_INT index);
CIC_EXTERN CIC_T_INT CheckACPort(CIC_T_INT channel_id);
CIC_EXTERN CIC_T_INT ReadBufPort(CIC_T_INT channel_id, CIC_T_UCHAR *buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT WriteBufPort(CIC_T_INT channel_id, CIC_T_UCHAR *buf, CIC_T_INT len);

#define MQ_RECEIVE(a, b, c) 	ReadPort(a, (CIC_T_UCHAR *)b, c)
#define MQ_SEND(a, b, c) 		WritePort(a, (CIC_T_UCHAR *)b, c)
#define MQ_AVAILABLE(a) 		Available(a)
#define AC_RECEIVE(a, b, c, d) 	ReadACPort(a, (CIC_T_UCHAR *)b, c, d)
#define AC_SEND(a, b, c, d) 	WriteACPort(a, (CIC_T_UCHAR *)b, c, d)
#define AC_CHECK(a) 			CheckACPort(a)
#define BUF_RECEIVE(a, b, c) 	ReadBufPort(a, (CIC_T_UCHAR *)b, c)
#define BUF_SEND(a, b, c) 		WriteBufPort(a, (CIC_T_UCHAR *)b, c)

CIC_EXTERN CIC_T_LONG 		Control_GetParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name);
CIC_EXTERN CIC_T_CIC_T_VOID Control_SetParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_LONG p_value);
CIC_EXTERN CIC_T_DOUBLE 	Control_GetParamFloat(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name);
CIC_EXTERN CIC_T_VOID 		Control_SetParamFloat(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_DOUBLE p_value);
CIC_EXTERN CIC_T_VOID 		Control_RunTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		Control_StopTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		Control_SuspendTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_namet);                             
CIC_EXTERN CIC_T_VOID 		Control_ResumeTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		Control_CallTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_INT 		Control_GetCurrentTimeBase(CIC_T_VOID);
CIC_EXTERN CIC_T_INT 		Control_SetTimer(CIC_T_CHAR* caller_task_name, CIC_T_UINT time_value, CIC_T_CHAR *time_unit);
CIC_EXTERN CIC_T_INT 		Control_GetTimerAlarmed(CIC_T_CHAR* caller_task_name, CIC_T_UINT timer_id);
CIC_EXTERN CIC_T_VOID 		Control_ResetTimer(CIC_T_CHAR* caller_task_name, CIC_T_UINT timer_id);
CIC_EXTERN CIC_T_VOID 		Control_ProgrammKill(CIC_T_VOID);
CIC_EXTERN CIC_T_VOID 		Control_ProgramStop(CIC_T_CHAR* caller_task_name);
CIC_EXTERN CIC_T_VOID 		Control_ExecuteTransition(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		Control_SetMTMParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_LONG p_value);
CIC_EXTERN CIC_T_VOID 		Control_SetMTMParamString(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_CHAR* p_value);
CIC_EXTERN CIC_T_CHAR* 		Control_GetMode(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_INT 		Control_CheckTaskState(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name); 


#define SYS_REQ_GET_PARAM_CIC_T_INT(a, b)		 Control_GetParamInt(TASK_NAME, a, b)
#define SYS_REQ_SET_PARAM_CIC_T_INT(a, b, c)	 Control_SetParamInt(TASK_NAME, a, b, c)
#define SYS_REQ_GET_PARAM_FLOAT(a, b)			 Control_GetParamFloat(TASK_NAME, a, b)
#define SYS_REQ_SET_PARAM_FLOAT(a, b, c)		 Control_SetParamFloat(TASK_NAME, a, b, c)
#define SYS_REQ_RUN_TASK(a)						 Control_RunTask(TASK_NAME, a)
#define SYS_REQ_STOP_TASK(a)					 Control_StopTask(TASK_NAME, a)
#define SYS_REQ_SUSPEND_TASK(a)					 SuspendTask(TASK_NAME, a)                                                     
#define SYS_REQ_RESUME_TASK(a)					 Control_ResumeTask(TASK_NAME, a)
#define SYS_REQ_CALL_TASK(a)					 Control_CallTask(TASK_NAME, a)
#define SYS_REQ_GET_CURRENT_TIME_BASE()			 Control_GetCurrentTimeBase()
#define SYS_REQ_SET_TIMER(a, b)					 Control_SetTimer(TASK_NAME, a, b)
#define SYS_REQ_GET_TIMER_ALARMED(a)			 Control_GetTimerAlarmed(TASK_NAME, a)
#define SYS_REQ_RESET_TIMER(a) 					 Control_ResetTimer(TASK_NAME, a)
#define SYS_REQ_KILL() 							 Control_ProgramKill()
#define SYS_REQ_STOP() 							 Control_ProgramStop(TASK_NAME)
#define SYS_REQ_EXECUTE_TRANSITION(a) 			 Control_ExecuteTransition(TASK_NAME, a)
#define SYS_REQ_SET_MTM_PARAM_CIC_T_INT(a, b, c) Control_SetMTMParamInt(TASK_NAME, a, b, c)
#define SYS_REQ_SET_MTM_PARAM_STRING(a, b, c) 	 Control_SetMTMParamString(TASK_NAME, a, b, c)
#define SYS_REQ_GET_MODE(a) 					 Control_GetMode(TASK_NAME, a)
#define SYS_REQ_CHECK_TASK_STATE(a) 			 Control_CheckTaskState(TASK_NAME, a)  

#endif /* __CIC_APIS_H__ */

