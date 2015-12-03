
typedef  enum {                                                 /* States for the control task.                         */
    CONTROL_IDLE = 0,
    CONTROL_DRIVE_FORWARD,
    CONTROL_DRIVING_FORWARD,
    CONTROL_TURN_LEFT,
    CONTROL_TURN_RIGHT,
    CONTROL_TURNING
} tControlTaskState;

typedef  enum {                                                 /* States for the PID task.                             */
    PID_IDLE = 0,
    PID_START,
    PID_RUNNING
} tPIDTaskState;

typedef  enum {                                                 /* Motor drive message types.                           */
    MSG_TYPE_MOTOR_DRIVE_START = 0,
    MSG_TYPE_MOTOR_STOP,
    MSG_TYPE_MOTOR_DRIVE
} tMotorMsgType;

typedef  enum {                                                 /* Motor drive message contents.                        */
    MOTOR_DRIVE_FORWARD = 0,
    MOTOR_DRIVE_REVERSE
} tMotorMsgContent;

                                                                /* Other defines.                                       */
#define INIT_DRIVE_TIME_WINDOW    (16 * OSCfg_TmrTaskRate_Hz)   /* 16 seconds.                                          */
#define MAX_RPM                 87
#define RIGHT_SIDE_SENSOR       SENSOR_A
#define LEFT_SIDE_SENSOR        SENSOR_A
#define RIGHT_SIDE_SENSOR_PORT  RIGHT_IR_SENSOR_A_PORT
#define RIGHT_SIDE_SENSOR_PIN   RIGHT_IR_SENSOR_A_PIN
#define LEFT_SIDE_SENSOR_PORT    LEFT_IR_SENSOR_A_PORT
#define LEFT_SIDE_SENSOR_PIN     LEFT_IR_SENSOR_A_PIN

                                                                /* Robot Control Task Flag Definitions.                 */
#define FLAG_PUSH_BUTTON          (OS_FLAGS)0x0001u
#define FLAG_RIGHT_BUMP_SENSOR    (OS_FLAGS)0x0002u
#define FLAG_LEFT_BUMP_SENSOR     (OS_FLAGS)0x0004u
#define FLAG_TIMER_EXPIRATION     (OS_FLAGS)0x0008u

                                                                /* Robot Motor PID Task Flag Definitions.               */
#define FLAG_PID_START            (OS_FLAGS)0x0001u
#define FLAG_PID_STOP             (OS_FLAGS)0x0002u
