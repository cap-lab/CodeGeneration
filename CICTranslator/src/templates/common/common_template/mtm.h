#ifndef __MTM_H__
#define __MTM_H__
#include "target_system_model.h"
#include "target_task_model.h"

#define CIC_ARRAYLEN(ARR) (CIC_SIZEOF(ARR)/CIC_SIZEOF(ARR[0]))

CIC_TYPEDEF CIC_T_STRUCT{
    CIC_T_INT id;
    CIC_T_CHAR* name;
}CIC_UT_MODEMAP;

CIC_TYPEDEF CIC_T_STRUCT{
    CIC_T_INT id;
    CIC_T_CHAR* name;
    CIC_T_INT value;
}CIC_UT_INT_VAR;

CIC_TYPEDEF CIC_T_STRUCT{
    CIC_T_INT id;
    CIC_T_CHAR* name;
    CIC_T_CHAR* value;
}CIC_UT_STR_VAR;

#endif /* __MTM_H__ */