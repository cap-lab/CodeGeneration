#ifndef __CIC_CONTROL_APIS_H__
#define __CIC_CONTROL_APIS_H__

CIC_TYPEDEF CIC_T_ENUM{ STATE_RUN, STATE_STOP, STATE_WAIT, STATE_END } CIC_UT_TASK_STATE;

CIC_EXTERN CIC_T_VOID 		SetThroughput(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* value, CIC_T_CHAR* unit);
CIC_EXTERN CIC_T_VOID 		SetDeadline(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* value, CIC_T_CHAR* unit);
CIC_EXTERN CIC_T_LONG 		GetParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name);
CIC_EXTERN CIC_T_VOID 		SetParamInt(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_LONG p_value);
CIC_EXTERN CIC_T_DOUBLE 	GetParamFloat(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name);
CIC_EXTERN CIC_T_VOID 		SetParamFloat(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name, CIC_T_CHAR* p_name, CIC_T_DOUBLE p_value);
CIC_EXTERN CIC_T_VOID 		RunCICTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		StopCICTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		SuspendCICTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_namet);                             
CIC_EXTERN CIC_T_VOID 		ResumeCICTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		CallCICTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_INT 		CheckTaskState(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name); 
CIC_EXTERN CIC_T_INT 		GetCurrentTimeBase(CIC_T_VOID);
CIC_EXTERN CIC_T_INT 		SetCICTimer(CIC_T_CHAR* caller_task_name, CIC_T_UINT time_value, CIC_T_CHAR *time_unit);
CIC_EXTERN CIC_T_INT 		GetCICTimerAlarmed(CIC_T_CHAR* caller_task_name, CIC_T_UINT timer_id);
CIC_EXTERN CIC_T_VOID 		ResetCICTimer(CIC_T_CHAR* caller_task_name, CIC_T_UINT timer_id);

#define SYS_REQ_SET_THROUGHPUT(a, b, c)			 SetThroughput(TASK_NAME, a, b, c)
#define SYS_REQ_SET_DEADLINE(a, b, c)			 SetDeadline(TASK_NAME, a, b, c)
#define SYS_REQ_GET_PARAM_INT(a, b)				 GetParamInt(TASK_NAME, a, b)
#define SYS_REQ_SET_PARAM_INT(a, b, c)	 		 SetParamInt(TASK_NAME, a, b, c)
#define SYS_REQ_GET_PARAM_FLOAT(a, b)			 GetParamFloat(TASK_NAME, a, b)
#define SYS_REQ_SET_PARAM_FLOAT(a, b, c)		 SetParamFloat(TASK_NAME, a, b, c)
#define SYS_REQ_RUN_TASK(a)						 RunCICTask(TASK_NAME, a)
#define SYS_REQ_STOP_TASK(a)					 StopCICTask(TASK_NAME, a)
#define SYS_REQ_SUSPEND_TASK(a)					 SuspendCICTask(TASK_NAME, a)                                                     
#define SYS_REQ_RESUME_TASK(a)					 ResumeCICTask(TASK_NAME, a)
#define SYS_REQ_CALL_TASK(a)					 CallCICTask(TASK_NAME, a)
#define SYS_REQ_CHECK_TASK_STATE(a) 			 CheckTaskState(TASK_NAME, a)
#define SYS_REQ_GET_CURRENT_TIME_BASE()			 GetCurrentTimeBase()
#define SYS_REQ_SET_TIMER(a, b)					 SetCICTimer(TASK_NAME, a, b)
#define SYS_REQ_GET_TIMER_ALARMED(a)			 GetCICTimerAlarmed(TASK_NAME, a)
#define SYS_REQ_RESET_TIMER(a) 					 ResetCICTimer(TASK_NAME, a)


#endif /* __CIC_CONTROL_APIS_H__ */

