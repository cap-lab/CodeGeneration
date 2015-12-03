#include "robot_kin.h"

    void MP::set_cur(double x, double y){
        cx=x;
        cy=y;
    }

    void MP::set_time(double time){
        t=time;
    }

    int MP::target(){
            int color;
            if ((abs(mark(0,0)-cx)+abs(mark(0,1)-cy))<0.5){
                color=1;

            }
            else if((abs(mark(1,0)-cx)+abs(mark(1,1)-cy))<0.5){
                color=2;

            }
            else if((abs(mark(2,0)-cx)+abs(mark(2,1)-cy))<0.5) {
                color=3;

            }
            else{
                color=0;

            }
            return color;
    }

    int MP::notify(){
            int warn;
            int yaw=(int)(headini);
            //cout<<"Yaw angle:"<<yaw<<endl;
            if (cx<0.0){
                switch(yaw){
                    case 0:
                    case 6:
                        warn=3; // move right
                        break;
                    case 4:
                    case -1:
                        warn=1; // move front
                        break;
                    case -4:
                    case 1:
                        warn=2; // move backward
                        break;
                    case 3:
                    case -3:
                        warn=4; // move left
                        break;
                }
            }
            else if(cx>5.0){
                switch(yaw){
                    case 0:
                    case 6:
                        warn=4; // move
                        break;
                    case 4:
                    case -1:
                        warn=2; // move
                        break;
                    case -4:
                    case 1:
                        warn=1; // move
                        break;
                    case 3:
                    case -3:
                        warn=3; // move
                        break;
                }

            }
            else if(cy<0.0) {
                switch(yaw){
                    case 0:
                    case 6:
                        warn=1; // move
                        break;
                    case 4:
                    case -1:
                        warn=4; // move
                        break;
                    case -4:
                    case 1:
                        warn=3; // move
                        break;
                    case 3:
                    case -3:
                        warn=2; // move
                        break;
                }

            }
            else if(cy>5.0) {
                switch(yaw){
                    case 0:
                    case 6:
                        warn=2; // move
                        break;
                    case 4:
                    case -1:
                        warn=3; // move
                        break;
                    case -4:
                    case 1:
                        warn=4; // move
                        break;
                    case 3:
                    case -3:
                        warn=1; // move
                        break;
                }
            }
            else
            {
                warn=0;
            }
            return warn;

    }

//    Matrix<double, 3, 2> MP::search(){
//
//            Matrix<double, 3, 2> out=MatrixXd::Zero(3,2);
//
//            if (Mode!=0){
//
//                if (sqrt(pow((mark(0,0)-cx),2)+pow((mark(0,1)-cy),2))<1.5){
//                    out.row(0)<<mark.row(0);
//                }
//                else if(sqrt(pow(mark(1,0)-cx,2)+pow((mark(1,1)-cy),2))<1.5){
//                    out.row(1)<<mark.row(1);
//                }
//                else if(sqrt(pow(mark(2,0)-cx,2)+pow((mark(2,1)-cy),2))<1.5) {
//                    out.row(2)<<mark.row(2);
//                }
//                else
//                    out=MatrixXd::Zero(3,2);
//            }
//
//            return out;
//    }

    void MP::initialize(Matrix<double, 2, 2> data, int path_case){

            ini<<data(0,0),data(0,1);
            fin<<data(1,0),data(1,1);

            Matrix<double, 2, 2> R;
            R<<cos(headini),-sin(headini),sin(headini),cos(headini);
            fino<<R*((fin-ini).transpose());
            cout<<"ini:"<<ini(0)<<","<<ini(1)<<endl;
            cout<<"fin:"<<fino(0)+ini(0)<<","<<fino(1)+ini(1)<<endl;
            cout<<"Heading:"<<headini<<endl;

        if (path_case==0)
        {
            ang_temp=atan2((fin(1)-ini(1)),(fin(0)-ini(0)))+headini;

            step=ceil(((fin-ini).norm())/Sfor);

            //cout<<((fin-ini).norm())<<endl;

            ang=ang_temp*MatrixXd::Ones(step,1);

            Sleft=abs((fin-ini).norm())-(step-1)*Sfor;

            //cout<<"ok:"<<endl;

            Sf.resize(step,1);
            Sf<<Sfor*MatrixXd::Ones(step-1,1),Sleft;

            Mode=path_case;

        }
        else{

            if (path_case==1)
                rotating=0.5*M_PI;
                else if(path_case==2)
                rotating=-0.5*M_PI;

            dang=M_PI/4;
            cout<<"ini:"<<ini<<endl;
            cout<<"fin:"<<fin<<endl;

            step=ceil(abs(rotating)/dang);
            Mode=path_case;

            if (rotating<0)
                rmode=-1;
            else
                rmode=1;

        }

        //cout<<"target angle:"<<ang_temp<<endl;
        Heading.resize(1,step+1);
        dHeading.resize(1,step);
        head.resize(1,step*(Time/dt)+1);
        //cout<<"headini:"<<headini<<dang;

        if (Mode==0){
            Heading<<headini*MatrixXd::Ones(1,step+1);
            head(0)=Heading(0);
        }
        else{
            Heading<<MatrixXd::Zero(1,step+1);
            //Heading<<headini*MatrixXd::Ones(1,step+1);
            for (int i=0;i<step+1;i++){
                Heading(i)=dang*i;
                if (i==(step)){
                    Heading(i)=dang*(step)+fmod(rotating,dang);}
            }
            if (rmode==-1)
                Heading=-Heading+headini*MatrixXd::Ones(1,step+1);
                else
                Heading=Heading+headini*MatrixXd::Ones(1,step+1);


            for (int i=0;i<step;i++){
                dHeading(i)=Heading(i+1)-Heading(i);
            }
            head(0)=Heading(0);
        }

        headini=Heading(step);
        if(headini>2*M_PI)
            headini=0;
            else if (headini<-2*M_PI)
                headini=0;




        //cout<<"rotation mode:"<<dHeading<<endl;
        // foot_start position in body coordinate
        xcb<<- size_x/2,size_x/2,-size_x/2,size_x/2;
        ycb<<  size_y/2,size_y/2,-size_y/2,-size_y/2;
        zcb<<0,0,0,0;


        //link initial condition for kinematics
        for (int i=0; i<4; i++){
            Si.col(i)<<xcb(i),ycb(i),zcb(i),1;
            temp<<pow(-1,i+1)*(e+d),0,-b-a,0;
            Fi.col(i)=Si.col(i)+temp.transpose();
            temp<<pow(-1,i+1)*(e),0,0,0;
            M1i.col(i)=Si.col(i)+temp.transpose();
            temp<<pow(-1,i+1)*(d),0,0,0;
            M2i.col(i)=M1i.col(i)+temp.transpose();
            temp<<0,0,a,0;
            M3i.col(i)=Fi.col(i)+temp.transpose();

            temp1<<-Si(0,i),-Si(1,i),-Si(2,i);
            e3<<0,0,1;
            ksi1.col(i)<<e3.cross(temp1),0,0,1;
            temp1<<-M1i(0,i),-M1i(1,i),-M1i(2,i);
            e3<<0,pow(-1,i+1),0;
            ksi2.col(i)<<e3.cross(temp1),0,pow(-1,i+1),0;
            temp1<<-M2i(0,i),-M2i(1,i),-M2i(2,i);
            ksi3.col(i)<<e3.cross(temp1),0,pow(-1,i+1),0;
        }
           // cout<<"Si:"<<ksi3<<endl;

           u0=e+d;
           v0=b+a;

           ac_standard=atan(size_x/(size_y+2*u0));
           ac1=ac_standard;
           ac2=M_PI-ac_standard;
           ac3=M_PI+ac_standard;
           ac4=2*M_PI-ac_standard;

           ci.resize(step+1,4);
           Cd.resize(step,4);
           cd.resize(step*(Time/dt)+1,4);
           xa.resize(step*(Time/dt)+1,4);
           ya.resize(step*(Time/dt)+1,4);
           za.resize(step*(Time/dt)+1,4);

           //cout<<"ok:"<<endl;

           Calc_order();
           trajectory();


    }


    void MP::Calc_order(){

        temp<<ini,v0,1;
        Matrix<double, 1, 4> temp2;
        temp2<<-u0,u0,-u0,u0;
        Matrix<double, 1, 4> temp3;
        temp3<<v0,v0,v0,v0;
        xb<<xcb + temp2,ycb,zcb-temp3;
        fb=Rz(head(0))*xb;
        xa.row(0)<<temp(0)*MatrixXd::Ones(1,4)  + fb.row(0);
        ya.row(0)<<temp(1)*MatrixXd::Ones(1,4)  + fb.row(1);
        za.row(0)<<temp(2)*MatrixXd::Ones(1,4)  + fb.row(2);

        ci.row(0)=temp;
        cd.row(0)=temp;
        order.resize(step,4);
        int iter=step;
        for (int i=0; i<iter; i++){
            if (Mode==1 || Mode==2 ){
                if (rmode==1 )
                    order.row(i)<<2,3,1,0;
                else
                    order.row(i)<<0,1,3,2;

            }
            else if (-Heading(i)+ang(i)<=ac1 & -Heading(i)+ang(i)>=0){
                order.row(i)<<2,3,0,1;
                maxtype=1;
            }
            else if (-Heading(i)+ang(i)<=M_PI/2 & -Heading(i)+ang(i)>=ac1){
                order.row(i)<<2,0,3,1;
                maxtype=2;}
            else if (-Heading(i)+ang(i)<=ac2 & -Heading(i)+ang(i)>M_PI/2){
                order.row(i)<<3,1,2,0;
                maxtype=2;}
            else if (-Heading(i)+ang(i)<=M_PI& -Heading(i)+ang(i)>ac2){
                order.row(i)<<3,2,1,0;
                maxtype=1;}
            else if (-Heading(i)+ang(i)<=ac3 &-Heading(i)+ang(i)>M_PI){
                  order.row(i)<<1,0,3,2;
                maxtype=1;}
            else if (-Heading(i)+ang(i)<=3*M_PI/2 & -Heading(i)+ang(i)>ac3){
                order.row(i)<<1,3,0,2;
                maxtype=2;}
            else if (-Heading(i)+ang(i)<=ac4 & -Heading(i)+ang(i)>3*M_PI/2){
                order.row(i)<<0,2,1,3;
                maxtype=2;}
            else
            {
                order.row(i)<<0,1,2,3;
                maxtype=1;

            }
        }
       //cout<<"temp"<<order<<endl;
    } // Calc_order

    void MP::trajectory(){

        float Stride=0;
        Matrix<double, 1, 4> temp2;
        Matrix<double, 4, 4> St1,St2,S,xb_t,ci_t,ci_t_pre;
        Matrix<double, 3, 3> R1,R2;

        timeiter=0;

        float dhead;

        int iter=step;
        //cout<<"iter"<<iter<<endl;
        for (int i=0; i<iter; i++){

            R1=Rz(Heading(i+1));
            R2=Rz(Heading(i));
            //cout<<"S:"<<R<<endl;
            if (Mode==0){
                Stride=Sf(i);
                temp2<<cos(ang(i)),sin(ang(i)),0,0;
                Cd.row(i)=ci.row(i)+Stride*temp2;
                ci.row(i+1)=Cd.row(i);
                Cd(i,2)=v0;}
            else{
                Cd.row(i)=ci.row(i);
                ci.row(i+1)=Cd.row(i);
                ci_t=MatrixXd::Ones(4,1)*ci.row(i+1);
                ci_t_pre=MatrixXd::Ones(4,1)*ci.row(i);
                St1<<R1.row(0),0,R1.row(1),0,R1.row(2),0,0,0,0,1;
                St2<<R2.row(0),0,R2.row(1),0,R2.row(2),0,0,0,0,1;
                xb_t<<xb,0,0,0,0;
                S=St1*xb_t+ci_t.transpose()-St2*xb_t-ci_t_pre.transpose();
                //S=St1;
            }

            //cout<<"S:"<<S<<endl;

            for (int j=0;j<(int)(Time/dt);++j){

                timeiter=timeiter+1;
                if (Mode==0){
                    cd.row(timeiter)=cd.row(timeiter-1)+(dt/Time)*Stride*temp2;
                    head(timeiter)=head(0);
                }
                else{
                    dhead=dt*dHeading(i)/Time;
                    head(timeiter)=head(timeiter-1)+dhead;
                    cd.row(timeiter)=cd.row(timeiter-1);
                }

                if ((j+1)<=(0.25*Time/dt))
                    swingorder=0;
                else if ((j+1)<=(0.5*Time/dt))
                    swingorder=1;
                else if ((j+1)<=(0.75*Time/dt))
                    swingorder=2;
                else
                    swingorder=3;

                if (Mode==0){
                    xa(timeiter,order(i,swingorder))=xa(timeiter-1,order(i,swingorder))+4.0*(dt/Time)*Stride*cos(ang(i));
                    ya(timeiter,order(i,swingorder))=ya(timeiter-1,order(i,swingorder))+4.0*(dt/Time)*Stride*sin(ang(i));
                    za(timeiter,order(i,swingorder))=-h*16*pow(dt/Time,2)*(j+1-(swingorder)*10)*(j+1-(swingorder)*10.0-Time/4.0/dt);
                }
                else{
                    xa(timeiter,order(i,swingorder))=xa(timeiter-1,order(i,swingorder))+4.0*(dt/Time)*S(0,order(i,swingorder));
                    ya(timeiter,order(i,swingorder))=ya(timeiter-1,order(i,swingorder))+4.0*(dt/Time)*S(1,order(i,swingorder));
                    za(timeiter,order(i,swingorder))=-h*16*pow(dt/Time,2)*(j+1-(swingorder)*10)*(j+1-(swingorder)*10.0-Time/4.0/dt);
                }
                for (int nonswing=0;nonswing<4;nonswing++){
                    if(order(i,swingorder)!=nonswing)
                    {
                        xa(timeiter,nonswing)=xa(timeiter-1,nonswing);
                        ya(timeiter,nonswing)=ya(timeiter-1,nonswing);
                        za(timeiter,nonswing)=za(timeiter-1,nonswing);
                    } // if
                }// for
            }
        }
       //cout<<"xa"<<cd<<endl;
       //cout<<"ok33"<<endl;
    } //trajectory

    void MP::kinematics(){
        Matrix<double, 1, 4> temp,cd_t;
        Matrix<double, 3, 3> R;
        Matrix<double, 4, 4> gb,footinv,f2,f3;
        float theta1,sum;
        int i;
        int sumtemp=0;

        i=t/0.02;


        //cout<<"time:"<<t<<endl;
        R=Rz(head(i));
        f2<<xa.row(i),ya.row(i),za.row(i),1,1,1,1;
        cd_t=cd.row(i);
        f3=f2-(cd_t.transpose())*MatrixXd::Ones(1,4);
        gb<<R.row(0),0,R.row(1),0,R.row(2),0,0,0,0,1;

        footinv=gb.lu().solve(f3)+(cd_t.transpose())*MatrixXd::Ones(1,4);

        //cout<<"footinv_test:"<<gb<<endl;

        dx=cd(i,0)*MatrixXd::Ones(1,4) + xcb-footinv.row(0);
        dy=cd(i,1)*MatrixXd::Ones(1,4) + ycb-footinv.row(1);
        temp=dx.array().square() +dy.array().square() ;
        u=temp.array().sqrt();
        v=cd(i,2)*MatrixXd::Ones(1,4) + zcb-za.row(i);

       // cout<<"u:"<<footinv<<endl;

        for (int j=0;j<4;j++){

            A1(0,j)=2*b*v(0,j)-2*a*b;
            A2(0,j)=2*b*u(0,j)-2*b*e;
            A3(0,j)=pow(a,2)+pow(b,2)+pow(e,2)-pow(d,2)-2*a*v(0,j)-2*e*u(0,j)+pow(v(0,j),2)+pow(u(0,j),2);

            B1(0,j)=2*d*v(0,j)-2*a*d;
            B2(0,j)=2*d*u(0,j)-2*d*e;
            B3(0,j)=pow(a,2)+pow(d,2)+pow(e,2)-pow(b,2)-2*a*v(0,j)-2*e*u(0,j)+pow(v(0,j),2)+pow(u(0,j),2);

            t1sol.col(j)<<(A1(0,j)+sqrt(pow(A1(0,j),2)+pow(A2(0,j),2)-pow(A3(0,j),2)))/(A2(0,j)+A3(0,j)),(A1(0,j)-sqrt(pow(A1(0,j),2)+pow(A2(0,j),2)-pow(A3(0,j),2)))/(A2(0,j)+A3(0,j));
            t2sol.col(j)<<(B1(0,j)+sqrt(pow(B1(0,j),2)+pow(B2(0,j),2)-pow(B3(0,j),2)))/(B2(0,j)+B3(0,j)),(B1(0,j)-sqrt(pow(B1(0,j),2)+pow(B2(0,j),2)-pow(B3(0,j),2)))/(B2(0,j)+B3(0,j));

            theta3.col(j)<<atan(t1sol(0,j))*2,atan(t1sol(1,j))*2;
            theta2.col(j)<<atan(t2sol(0,j))*2,atan(t2sol(1,j))*2;
            theta1=atan(dy(j)/dx(j));
            sumtemp=100;
            for (int m=0;m<2;m++){
                for (int n=0;n<2;n++){
                    if(abs(cos(theta2(n,j))*d+b*cos(theta3(m,j))+e-u(j))<0.125 & abs(sin(theta2(n,j))*d+b*sin(theta3(m,0))+a-v(j))<0.125){
                        sum=abs(cos(theta2(n,j))*d+b*cos(theta3(m,j))+e-u(j))+abs(sin(theta2(n,j))*d+b*sin(theta3(m,0))+a-v(j));
                        if (theta3(m,j) >= theta2(n,j) & sum <= sumtemp) {
                            theta.col(j)<<theta1,theta2(n,j),theta3(m,j);
                        }
                        sumtemp=sum;
                    }
                }
            }
        }

        for (int j=0;j<4;j++){

            phi(j)=theta(0,j);
            psi(j)=theta(1,j);
            gamma(j)=theta(2,j)-theta(1,j)-M_PI/2;

            g1=exp_k(ksi1.col(j),phi(j));
            g2=exp_k(ksi2.col(j),psi(j));
            g3=exp_k(ksi3.col(j),gamma(j));

            Matrix<double, 4, 1> test,test1;
            test<<cd(i,0),cd(i,1),cd(i,2),0;
            test1<<0,0,a,0;
            ps.col(j)=gb*g1*Si.col(j)+test;
            pm1.col(j)=gb*g1*M1i.col(j)+test;
            pm2.col(j)=gb*g1*g2*M2i.col(j)+test;
            pm3.col(j)=gb*g1*g2*g3*M3i.col(j)+test;
            pf.col(j)=pm3.col(j)-test1;

            p_lf.row(j)<<ps(j,0),pm1(j,0),pm2(j,0),pm3(j,0),pf(j,0);
            p_rf.row(j)<<ps(j,1),pm1(j,1),pm2(j,1),pm3(j,1),pf(j,1);
            p_lr.row(j)<<ps(j,2),pm1(j,2),pm2(j,2),pm3(j,2),pf(j,2);
            p_rr.row(j)<<ps(j,3),pm1(j,3),pm2(j,3),pm3(j,3),pf(j,3);

        }

   //cout<<"P:"<<ps<<endl;

    }

    Matrix<double, 4, 4> MP::exp_k(Matrix<double, 6, 1> ksi,float theta){

        float test;
        Matrix<double, 4, 4> g;
        Matrix<double, 3, 1> v,w,p;
        Matrix<double, 3, 3> hat,R;

        v<<ksi(0,0),ksi(1,0),ksi(2,0);
        w<<ksi(3,0),ksi(4,0),ksi(5,0);
        hat<<0,-w(2,0),w(1,0),w(2,0),0,-w(0,0),-w(1,0),w(0,0),0;
        R=MatrixXd::Identity(3,3)+sin(theta)*hat+(1-cos(theta))*hat*hat; // rodrigue's formula
        p=(MatrixXd::Identity(3,3)-R)*(w.cross(v))+theta*(w(0)*v(0)+w(1)*v(1)+w(2)*v(2))*w;
        g<<R.row(0),p(0),R.row(1),p(1),R.row(2),p(2),0,0,0,1;
        return g;

    }

    Matrix<double, 3, 3> MP::Rz(float theta){
        Matrix<double, 3, 3> R;
        R<<cos(theta),-sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;
        if (theta==M_PI/2 || theta==-M_PI/2)
            R<<0,-sin(theta), 0, sin(theta), 0, 0, 0, 0, 1;

        return R;
    }


