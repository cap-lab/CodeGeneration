#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <math.h>
#include <eigen3/Eigen/LU>
#include <stdlib.h>
#include <iostream>

using namespace Eigen;
using namespace std;

extern Matrix<double, 3, 2> mark;

class MP
{
    public:
        void kinematics();
       // void DISPLAY();
        int target();
        int notify();
        void set_cur(double x, double y);
        void set_time(double time);
        //Matrix<double, 3, 2> search();
        void initialize(Matrix<double, 2, 2> data, int path_case);
        //void generate_map();
        Matrix<double, 1, 2> cur_pos(){Matrix<double, 1, 2> c;c<<cx,cy; return c;};


        int step;
        float Time=0.8, dt=0.02;

        Matrix<double, 4, 5> p_lf,p_rf,p_lr,p_rr;
        Matrix<double, 4, 4> ps;
        int n=2; //boundary plot
        double t=0;
    protected:
        void Calc_order();
        void trajectory();

        Matrix<double, 4, 4> exp_k(Matrix<double, 6, 1> ksi,float theta);
        Matrix<double, 3, 3> Rz(float theta);


        int ms=20; //mash grid number

        double cx, cy;

        int timetotal=0;

        float a=0.043, b=0.2, d=0.155, e=0.045; //link size
        float size_x=0.202, size_y=0.402; //body size

//        dvec bx1,by1,bz1;
//        dvec bx2,by2,bz2;
//        dvec x,y,z;
//        dvec x1,y1,z1;
//        dvec x2,y2,z2;
//        dvec x3,y3,z3;
//        dvec x4,y4,z4;
//        dvec xmr,ymr,zmr;
//        dvec xmg,ymg,zmg;
//        dvec xmb,ymb,zmb;

        float Sfor=0.25;
        float Sleft,rotating, dang;
        int Mode,rmode;

        Matrix<double, Dynamic, Dynamic> order,Cd,ci,cd,xa,ya,za,Heading,dHeading,head;
        Matrix<double, 1, 2> ini;
        Matrix<double, 1, 2> fin;
        Matrix<double, 2, 1> fino;
        Matrix<double, 2, 4> t1sol,t2sol,theta3,theta2;
        Matrix<double, 3, 4> theta;

        Matrix<double, 6, 4> ksi1,ksi2,ksi3;
        Matrix<double, 1, 4> phi,psi,gamma, A1, A2, A3, B1, B2, B3,xcb,ycb,zcb,temp;
        Matrix<double, 1, 4> dx,dy,u,v;
        Matrix<double, 4, 4> Si, Fi, M1i, M2i, M3i,pf, pm1, pm2, pm3;

        Matrix<double, 4, 4> g1,g2,g3;

        Matrix<double, 3, 1> e3={0,0,1};
        Matrix<double, 3, 1> temp1;
        Matrix<double, 3, 4> fb,xb;

        float ang_temp,u0,v0,ac_standard,ac1,ac2,ac3,ac4;

        int maxtype,swingorder;
        int timeiter=0;

        int Timetotal;
        float h=0.15; // swing height
        float headini=0.0;

        VectorXd ang,Sf;
};

