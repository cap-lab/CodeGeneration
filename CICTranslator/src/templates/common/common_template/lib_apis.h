#ifndef __LIB_PORT_H__
#define __LIB_PORT_H__

CIC_EXTERN CIC_T_INT InitLibPort(CIC_T_INT task_id, CIC_T_CHAR op);
CIC_EXTERN CIC_T_INT ReadLibPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT WriteLibPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT LibAvailable(CIC_T_INT channel_id);

#define LIB_RECEIVE(a, b, c) ReadLibPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c)
#define LIB_SEND(a, b, c) WriteLibPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c)
#define LIB_CHECK(a) LibAvailable(a)

#endif /* __LIB_PORT_H__ */

