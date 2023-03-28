#include"matrix.hpp"
#include"mfunc.hpp"
#include<iostream>
#include<cstdio>
#include<random>
#include<time.h>
//#include<omp.h>
#define M 3
#define K 10
#define N 3333
using namespace std;
float data[M*K];
float data2[M*M];
float data3[K*N];
float data4[M*K];
int main(){
    default_random_engine e;
	uniform_real_distribution<double> u(-1.2,3.5);
    // test
    cout<<endl;
    cout<<"---------------------------------------the original data in double------------------------------------"<<endl;
    //M1 original matrix
    matrix<double> M1 ( M, M );
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            M1.set_val(i,j,u(e));
        }
    }
    cout<<"The original matrix is:"<<endl;
    cout<<M1<<endl;
    cout<<"-----------------test the sum and average of matrix-------------------"<<endl;
    cout<<"test the sum of row 2:";
    double sum=0;
    exception_t e02=MSUM_ROW(M1,2,sum);
    cout<<sum<<endl;
    cout<<"test the average of row 2:";
    double average=0;
    exception_t e03=MAVG_ROW(M1,2,average);
    cout<<average<<endl;
    cout<<"test the sum of column 2:";
    exception_t e04=MSUM_COL(M1,2,sum);
    cout<<sum<<endl;
    cout<<"test the average of column 2:";
    exception_t e05=MAVG_COL(M1,2,average);
    cout<<average<<endl;
    cout<<"test the sum of matrix:";
    exception_t e06=MSUM(M1,sum);
    cout<<sum<<endl;
    cout<<"test the average of matrix:";
    exception_t e07=MAVG(M1,average);
    cout<<average<<endl;

    cout<<"------------------------test the determinant--------------------------"<<endl;
    //rst determinant
    double rst=0;
    exception_t e01=MDET(M1,rst);
    cout<<"The determinant is "<<rst<<endl<<endl;

    cout<<"---------------test inverse matrix and adjoint matrix-----------------"<<endl;
    //M2 is the inverse matrix of M1
    matrix<double> MIN ( M, M );
    //M3 is the adjoint matrix of M1
    matrix<double> MAD ( M, M );
    exception_t e0=MINVER(M1,MIN);
    exception_t e1=MADJ(M1,MAD,M);
    cout<<"The inverse matrix is:"<<endl;
    cout<<MIN<<endl;
    cout<<"The adjoint matrix is:"<<endl;
    cout<<MAD<<endl;
    cout<<"--------------------test eigenvector and eigenvalue--------------------"<<endl;
        for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            M1.set_val(i,j,M1.get_val(j,i));
        }
    }
    cout<<"The M1 is changed as a real symmetric matrix:"<<endl;
    cout<<M1<<endl;
    matrix<double> MVECTOR ( M, M );
    matrix<double> MVALUE ( M, 1 );
    exception_t e3=MEGVLCT(M1,MVALUE,MVECTOR,0.01,10);
    cout<<"The eigenvalue is:"<<endl;
    cout<<MVALUE<<endl;
    cout<<"The eigenvector is:"<<endl;
    cout<<MVECTOR<<endl;


    cout<<endl;
    cout<<"---------------------------------------the original data in interger------------------------------------"<<endl;
    //M5 original matrix
    matrix<int> M5 ( M, M );
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            M5.set_val(i,j,rand()%10-1);
        }
    }
    cout<<"The original matrix is:"<<endl;
    cout<<M5<<endl;
    cout<<"-----------------test the sum and average of matrix-------------------"<<endl;
    cout<<"test the sum of row 2:";
    int sum2=0;
    exception_t e02_=MSUM_ROW(M5,2,sum2);
    cout<<sum2<<endl;
    cout<<"test the average of row 2:";
    int average2=0;
    exception_t e03_=MAVG_ROW(M5,2,average2);
    cout<<average2<<endl;
    cout<<"test the sum of column 2:";
    exception_t e04_=MSUM_COL(M5,2,sum2);
    cout<<sum2<<endl;
    cout<<"test the average of column 2:";
    exception_t e05_=MAVG_COL(M5,2,average2);
    cout<<average2<<endl;
    cout<<"test the sum of matrix:";
    exception_t e06_=MSUM(M5,sum2);
    cout<<sum2<<endl;
    cout<<"test the average of matrix:";
    exception_t e07_=MAVG(M5,average2);
    cout<<average2<<endl;

    cout<<"------------------------test the determinant--------------------------"<<endl;
    //rst determinant
    int rst2=0;
    exception_t e01_=MDET(M5,rst2);
    cout<<"The determinant is "<<rst2<<endl<<endl;
    cout<<"---------------test inverse matrix and adjoint matrix-----------------"<<endl;
    //MIN2 is the inverse matrix of M1
    matrix<int> MIN2 ( M, M );
    //MAD2 is the adjoint matrix of M1
    matrix<int> MAD2 ( M, M );
    exception_t e0_=MINVER(M5,MIN2);
    exception_t e1_=MADJ(M5,MAD2,M);
    cout<<"The inverse matrix is(almost inverse element is smaller than 1):"<<endl;
    cout<<MIN2<<endl;
    cout<<"The adjoint matrix is:"<<endl;
    cout<<MAD2<<endl;
}
