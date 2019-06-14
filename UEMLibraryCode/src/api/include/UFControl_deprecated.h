/*
 * UFControl_deprecated.h
 *
 *  Created on: 2017. 8. 13.
 *      Author: jej
 */

#ifndef SRC_API_INCLUDE_UFCONTROL_DEPRECATED_H_
#define SRC_API_INCLUDE_UFCONTROL_DEPRECATED_H_

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief (Deprecated) Finish task execution.
 *
 * It is terminated after The job currently being run is done.
 * If a subgraph is included, it is terminated after all the jobs of the current subgraph are completed.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to be terminated.
 *
 * @return
 *
 *
 */
void SYS_REQ_END_TASK(int nCallerTaskId, char *pszTaskName);

/**
 * @brief (Deprecated) Execute task.
 *
 * The task keeps running all the time.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to be ran.
 *
 * @return
 */
void SYS_REQ_RUN_TASK(int nCallerTaskId, char *pszTaskName);

/**
 * @brief (Deprecated) Stop task execution.
 *
 * The task could be continued by @ref UFControl_RunTask.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to be stopped.
 *
 * @return
 *
 *
 */
void SYS_REQ_STOP_TASK(int nCallerTaskId, char *pszTaskName);

/**
 * @brief (Deprecated) Execute a task only once.
 *
 * TASK_INIT and TASK_WRAPUP are also called when executing.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to be executed.
 *
 * @return
 */
void SYS_REQ_CALL_TASK(int nCallerTaskId, char *pszTaskName);

#ifndef API_LITE

/**
 * @brief (Deprecated) Suspend a task.
 *
 * The task could be continued by @ref UFControl_ResumeTask.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to be suspended.
 *
 * @return
 */
void SYS_REQ_SUSPEND_TASK(int nCallerTaskId, char *pszTaskName);

/**
 * @brief (Deprecated) Resume suspended task.
 *
 * @param nCallerTaskId id of caller task.
 * @param pszTaskName task to be resumed.
 *
 * @return
 */
void SYS_REQ_RESUME_TASK(int nCallerTaskId, char *pszTaskName);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SRC_API_INCLUDE_UFCONTROL_DEPRECATED_H_ */
