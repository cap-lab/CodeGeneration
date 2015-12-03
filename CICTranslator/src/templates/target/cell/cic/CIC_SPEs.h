#ifndef __LOCKING_SPES_H__
#define __LOCKING_SPES_H__

#define NUM_SPE (6)

typedef uint32_t CIC_SPEs_status;

extern CIC_SPEs_status CIC_SPEs_make_request(int count, ...);
extern bool CIC_SPEs_lock(CIC_SPEs_status request);
extern bool CIC_SPEs_unlock(CIC_SPEs_status request);

#endif /* __LOCKING_SPES_H__ */

