#ifndef __CIC_CMD_H__
#define __CIC_CMD_H__

typedef enum
{
    CIC_CMD_TASK_SWITCH,

    CIC_CMD_NOTIFY_GO_END,
    CIC_CMD_NOTIFY_WRAPUP_END,

    CIC_CMD_AC_RECEIVE,
    CIC_CMD_AC_RECEIVE_FINISH,
    CIC_CMD_AC_SEND,
    CIC_CMD_AC_SEND_FINISH,

    CIC_CMD_AC_AVAILABLE,
    CIC_CMD_AC_CHECK,

    CIC_CMD_TERMINATE = 0xffffffff,

} CIC_CMD;

#endif /* __CIC_CMD_H__ */
