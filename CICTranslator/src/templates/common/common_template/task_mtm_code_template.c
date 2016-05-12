#define MTM_INITIALIZE CIC_T_VOID ##TASK_NAME_Initialize(CIC_T_VOID)    
#define GET_CURRENT_MODE_NAME CIC_T_CHAR* ##TASK_NAME_GetCurrentModeName(CIC_T_CHAR *task_name)                             
#define GET_CURRENT_MODE_ID CIC_T_INT ##TASK_NAME_GetCurrentModeID(CIC_T_CHAR *task_name)                                                            
#define GET_MODE_NAME CIC_T_CHAR* ##TASK_NAME_GetModeName(CIC_T_INT id)                                       
#define GET_VARIABLE_INT CIC_T_INT ##TASK_NAME_GetVariableInt(CIC_T_CHAR* name)                               
#define SET_VARIABLE_INT CIC_T_VOID ##TASK_NAME_SetVariableInt(CIC_T_CHAR* name, CIC_T_INT value)                   
#define GET_VARIABLE_STRING CIC_T_CHAR* ##TASK_NAME_GetVariableString(CIC_T_CHAR* name)                       
#define SET_VARIABLE_STRING CIC_T_VOID ##TASK_NAME_SetVariableString(CIC_T_CHAR* name, CIC_T_CHAR* value)           
#define TRANSITION CIC_T_BOOL ##TASK_NAME_Transition()
#define UPDATE_CURRENT_MODE CIC_T_VOID ##TASK_NAME_UpdateCurrentMode(CIC_T_CHAR *task_name)

#define GET_TASK_ITER_COUNT CIC_T_INT ##TASK_NAME_GetTaskIterCount(CIC_T_CHAR* task_name)
#define GET_TASK_ITER_COUNT_FROM_MODE_NAME CIC_T_INT ##TASK_NAME_GetTaskIterCountFromModeName(CIC_T_CHAR* task_name, CIC_T_CHAR* mode_name)
#define GET_TASK_REPEAT_COUNT CIC_T_INT ##TASK_NAME_GetTaskRepeatCount(CIC_T_CHAR* task_name, CIC_T_INT run_count)

##CIC_INCLUDE
