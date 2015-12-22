#ifndef __CIC_BASIC_CONTROL_APIS_H__
#define __CIC_BASIC_CONTROL_APIS_H__

CIC_EXTERN CIC_T_VOID 		EndTask(CIC_T_CHAR* caller_task_name, CIC_T_CHAR* t_name);
CIC_EXTERN CIC_T_VOID 		ProgrammKill(CIC_T_VOID);
CIC_EXTERN CIC_T_VOID 		ProgramStop(CIC_T_CHAR* caller_task_name);

#define SYS_REQ_END_TASK(a)						 EndTask(TASK_NAME, a)
#define SYS_REQ_KILL() 							 ProgramKill()
#define SYS_REQ_STOP() 							 ProgramStop(TASK_NAME)

#endif /* __CIC_BASIC_CONTROL_APIS_H__ */

