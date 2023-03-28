#include"matrix.hpp"
#include"mfunc.hpp"
#include<iostream>
#include<cstdio>
#include<omp.h>
#include<complex>
#define M 102
#define K 107
#define N 102
using namespace std;
complex<float> data[M*K];
complex<float> data3[K*N];
complex<float> data4[M*N];
int main(){

    srand((unsigned)time(NULL));
    for(int i = 0 ; i < M ; i ++)
        for(int j = 0 ; j < K ; j ++)
            data[i*K+j] = complex<float>(rand()%10,rand()%10);
    for(int i = 0 ; i < K ; i ++)
        for(int j = 0 ; j < N ; j ++)
            data3[i*N+j] = complex<float>(rand()%10,rand()%10);

    matrix<complex<float>>M1(data,M,K,matrix<complex<float>>::SPARSE_TAG);
    matrix<complex<float>>M2(data3,K,N,matrix<complex<float>>::SPARSE_TAG);
    matrix<complex<float>>M3;

    double t1 = omp_get_wtime();
    MMULT(M1,M2,M3);
    cout<<omp_get_wtime()-t1<<endl;
    
    t1 = omp_get_wtime();
    for(int i = 0 ; i < M ; i ++){
        for(int j = 0 ; j < N ; j ++){
            for(int k = 0 ; k < K ; k ++)
                data4[i*N+j]+=data[i*K+k]*data3[k*N+j];
        }
    }
    cout<<omp_get_wtime()-t1<<endl;

    t1 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0 ; i < M ; i ++)
        for(int j = 0 ; j < N ; j ++)
            if(data4[i*N+j] != M3.get_val(i,j))
                cout<<"WA"<<endl;
    cout<<omp_get_wtime()-t1<<endl;



}