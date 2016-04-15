#ifndef __CIC_COMM_APIS_H__
#define __CIC_COMM_APIS_H__

CIC_EXTERN CIC_T_INT InitTaskPort(CIC_T_INT task_id, CIC_CONST CIC_T_CHAR* port_name);
CIC_EXTERN CIC_T_INT GetChannelIndexFromChannelId(CIC_T_INT channel_id);
CIC_EXTERN CIC_T_INT ReadPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT WritePort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT Available(CIC_T_INT channel_id);
CIC_EXTERN CIC_T_INT ReadACPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len, CIC_T_INT index);
CIC_EXTERN CIC_T_INT WriteACPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len, CIC_T_INT index);
CIC_EXTERN CIC_T_INT CheckACPort(CIC_T_INT channel_id);
CIC_EXTERN CIC_T_INT ReadBufPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len);
CIC_EXTERN CIC_T_INT WriteBufPort(CIC_T_INT channel_id, CIC_UT_BUFFER_DATA_TYPE buf, CIC_T_INT len);

#define PORT_INITIALIZE(a, b)	InitTaskPort(a, b)
#define MQ_RECEIVE(a, b, c) 	ReadPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c)
#define MQ_SEND(a, b, c) 		WritePort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c)
#define MQ_AVAILABLE(a) 		Available(a)
#define AC_RECEIVE(a, b, c, d) 	ReadACPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c, d)
#define AC_SEND(a, b, c, d) 	WriteACPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c, d)
#define AC_CHECK(a) 			CheckACPort(a)
#define BUF_RECEIVE(a, b, c) 	ReadBufPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c)
#define BUF_SEND(a, b, c) 		WriteBufPort(a, (CIC_UT_BUFFER_DATA_TYPE) b, c)

#endif /* __CIC_COMM_APIS_H__ */

