#ifndef __CIC_ROBOT_H__
#define __CIC_ROBOT_H__

extern int go_forward();
extern int stop();
extern int go_backward();
extern int turn(int direction, int angle);
extern int locate_point(char* xyz, int xyzSize);
//extern int int get_vision_img(simxInt* width, simxInt* height,simxUChar* image)
extern int meet_obstacle();

#define ROBOT_GO_FORWARD() go_forward()
#define ROBOT_STOP() stop()
#define ROBOT_GO_BACKWARD() go_backward()
#define ROBOT_TURN_LEFT(a) turn(0, a)
#define ROBOT_TURN_RIGHT(a) turn(1, a)

//���Ŀ� x,y,z�� ������ �Ѱ��൵ ���� �� ������... float, int�� ��������?? ������ float���� �ϸ� ���� ��������..?
#define ROBOT_LOCATE_POINT(a, b) locate_point(a, b)

//#define ROBOT_GET_VISION_IMG(a, b, c) get_vision_img(a, b, c)
#define ROBOT_MEET_OBSTACLE() meet_obstacle()

#endif /* __CIC_ROBOT_H__ */
