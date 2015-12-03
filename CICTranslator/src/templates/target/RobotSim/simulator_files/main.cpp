using namespace std;

#include "robot_kin.h"
#include <unistd.h>
#include <eigen3/Eigen/Dense>
int is_run=1;
int done=0;
double t=0;// tim


MP mp;
void display(){mp.display(); }
void reshape(int w,int h){ mp.reshape(w,h); }
void idle( void ){

    if(is_run){ t+=0.02; }
    glutPostRedisplay();
    usleep(20000);
}
void mouse(int button, int state, int x, int y ){ mp.mouse(button,state,x,y); }
void motion(int x, int y ){mp.motion(x,y); }
void passive(int x, int y ){mp.passivemotion(x,y); }
void keyboard(unsigned char key, int x, int y){
    mp.keyboard(key, x, y);
    if(key=='r'){ if(is_run==0){is_run=1;}else{is_run=0;}}
    if(key=='d' && done==1) //move right
    {
        is_run==0;
        int path_case;
        path_case=0;
        Matrix<double, 1, 2> cur;
        cur<<mp.cur_pos();
        cout<<cur<<endl;
        Matrix<double, 2, 2> data;
        data << cur(0), cur(1), cur(0)+0.5, cur(1) ;
        mp.initialize(data,path_case);
        t=0.02;
        is_run==1;
    }

    if(key=='w'&& done==1) //move forward
    {
        is_run==0;
        int path_case;
        path_case=0;
        Matrix<double, 1, 2> cur;
        cur<<mp.cur_pos();
        cout<<cur<<endl;
        Matrix<double, 2, 2> data;
        data << cur(0), cur(1), cur(0), cur(1)+0.5 ;
        mp.initialize(data,path_case);
        t=0.02;
        is_run==1;
    }

    if(key=='a'&& done==1) //move left
    {
        is_run==0;
        int path_case;
        path_case=0;
        Matrix<double, 1, 2> cur;
        cur<<mp.cur_pos();
        cout<<cur<<endl;
        Matrix<double, 2, 2> data;
        data << cur(0), cur(1), cur(0)-0.5, cur(1) ;
        mp.initialize(data,path_case);
        t=0.02;
        is_run==1;
    }

    if(key=='s'&& done==1) //move backward
        {
        is_run==0;
        int path_case;
        path_case=0;
        Matrix<double, 1, 2> cur;
        cur<<mp.cur_pos();
        cout<<cur<<endl;
        Matrix<double, 2, 2> data;
        data << cur(0), cur(1), cur(0), cur(1)-0.5;
        mp.initialize(data,path_case);
        t=0.02;
        is_run==1;
    }

    if(key=='z'&& done==1) //rotate ccw
    {
        is_run==0;
        int path_case;
        path_case=1;
        Matrix<double, 1, 2> cur;
        cur<<mp.cur_pos();
        cout<<cur<<endl;
        Matrix<double, 2, 2> data;
        data << cur(0), cur(1), cur(0), cur(1);
        mp.initialize(data,path_case);
        t=0.02;
        is_run==1;
    }

    if(key=='x'&& done==1) //rotate cw
    {
        is_run==0;
        int path_case;
        path_case=2;
        Matrix<double, 1, 2> cur;
        cur<<mp.cur_pos();
        cout<<cur<<endl;
        Matrix<double, 2, 2> data;
        data << cur(0), cur(1), cur(0), cur(1);
        mp.initialize(data,path_case);
        t=0.02;
        is_run==1;
    }


}
int main(int argc, char* argv[]){

    glutInit(&argc, argv);
    glutCreateWindow(300,300,1200,900);

    Matrix<double, 2, 2> data;
    int path_case=0;
    data << 0.0, 0.0, 0.5, 0.0 ;
    mp.initialize(data,path_case);
    mp.generate_map();

    glutDisplayFunc( display );
    glutReshapeFunc( reshape );
    glutIdleFunc( idle );
    glutMotionFunc( motion );
    glutMouseFunc( mouse );
    glutPassiveMotionFunc(passive);
    glutKeyboardFunc( keyboard );
    glutMainLoop();
    return 0;
}
