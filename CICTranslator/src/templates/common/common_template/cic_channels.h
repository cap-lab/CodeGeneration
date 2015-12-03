#ifndef __CIC_CHANNELS_H__
#define __CIC_CHANNELS_H__

CIC_TYPEDEF CIC_T_ENUM {
    CIC_UT_CHANNEL_TYPE_NORMAL,
    CIC_UT_CHANNEL_TYPE_BUFFER,
    CIC_UT_CHANNEL_TYPE_ARRAY_CHANNEL,
} CIC_UT_CHANNEL_TYPE;

CIC_TYPEDEF CIC_T_STRUCT _AC_AVAIL_LIST{
    CIC_T_INT avail_index;
    CIC_T_STRUCT _AC_AVAIL_LIST* prev;
    CIC_T_STRUCT _AC_AVAIL_LIST* next;
} CIC_UT_AC_AVAIL_LIST;

CIC_TYPEDEF CIC_T_STRUCT {
    CIC_UT_AC_AVAIL_LIST *avail_node;
    CIC_T_INT used;
    CIC_UT_BUFFER_DATA_TYPE buf;
} CIC_UT_AC_DATA;

CIC_TYPEDEF CIC_T_STRUCT {
    CIC_T_INT channel_id;
    CIC_T_INT next_channel_id;
    CIC_UT_CHANNEL_TYPE type;
    CIC_UT_BUFFER_DATA_TYPE buf;
    CIC_UT_BUFFER_DATA_TYPE start;
    CIC_UT_BUFFER_DATA_TYPE end;
    CIC_T_INT max_size;
    CIC_T_INT cur_size;
    CIC_UT_AC_DATA *head;

    CIC_UT_AC_AVAIL_LIST *avail_index_start;
    CIC_UT_AC_AVAIL_LIST *avail_index_end;

    CIC_T_INT init_data;
    CIC_T_INT sample_size;
    CIC_T_CHAR *sample_type;
    CIC_T_BOOL request_read;
    CIC_T_BOOL request_write;
    CIC_T_INT source_port;
    CIC_T_INT sink_port;
    CIC_T_BOOL is_full;
    CIC_T_BOOL is_watch;
    CIC_T_BOOL is_break;

    CIC_T_MUTEX mutex;
    CIC_T_COND cond;
} CIC_UT_CHANNEL;

#endif /* __CIC_CHANNELS_H__ */
