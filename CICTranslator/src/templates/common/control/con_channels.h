CIC_TYPEDEF CIC_T_STRUCT{
    CHANNEL_TYPE type;
    CIC_T_UCHAR *buf;
    CIC_T_UCHAR *start;
    CIC_T_UCHAR *end;
    CIC_T_INT max_size;
    CIC_T_INT cur_size;
    CIC_UT_AC_DATA *head;

    CIC_UT_AC_AVAIL_LIST *avail_index_start;
    CIC_UT_AC_AVAIL_LIST *avail_index_end;

    CIC_T_MUTEX mutex;
    CIC_T_COND cond;
    CIC_T_INT initData;
    CIC_T_INT sampleSize;
    CIC_T_BOOL request_read;
    CIC_T_BOOL request_write;
    CIC_VOLATILE CIC_T_INT used;
} CIC_UT_CON_CHANNEL_UNIT;

CIC_TYPEDEF CIC_T_STRUCT{
    CIC_T_INT channel_id;
    CIC_T_INT used;

    CON_CHANNEL_UNIT send_channel;
    CON_CHANNEL_UNIT recv_channel;

    CIC_T_THREAD c_mutex;
    CIC_T_COND c_cond;
} CIC_UT_CON_CHANNEL;

