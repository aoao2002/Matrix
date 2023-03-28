#pragma once
#include"matrix.hpp"
#include<complex>
#include<immintrin.h>


/*---------------- sparse matrix operation ----------------*/


/*
    get all data in a sparse matrix and store the data in `*rst`
    remember to allocate mem first
*/
template<typename T>
exception_t get_sparse_data(const matrix<T>& A,T*& rst){
    if(!A.is_sparse()) return MATRIX_NOT_SPARSE;
    if(rst != NULL) {
        exception_handler::puts_err("rst Invalid , Only rst == NULL is allowed.");
        return MATRIX_DATA_INVALID;
    }
    rst = new T[A.get_rownum()*A.get_colnum()];
    memset(rst,0,sizeof(T)*A.get_colnum()*A.get_rownum());
    const std::vector<triplet<T>>& sparse_data = A.get_sparse_data();

    for(triplet<T> t : sparse_data)
        rst[t.getrow()*A.get_colnum()+t.getcol()] = t.getdata();

    return OP_SUCCESS;
}



/*---------------- Matrix vector arithmetic operation ----------------*/



/*
    Cross-type Matrix operation
    the user him/herself shoud check the validity of the 
    arithmetic operation and type conversion
    its not the designer's work to ensure that
    Matix-Matrix(Vector-Vector) multiplication
    return exception and store the result in `rst`
*/
template<
    typename T1,
    typename T2,
    typename T3
>
exception_t MMULT(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst){
    if(A.get_colnum() != B.get_rownum()) return MATRIX_SIZE_INVALID;

    T3* rst_arr = new T3[A.get_rownum()*B.get_colnum()];
    T3* A_dense_data = NULL;
    T3* B_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    if(B.is_sparse())
        get_sparse_data(B,B_dense_data);
    else B_dense_data = B.dense_data;

    T3* B_TRANS = new T3[B.get_colnum()*B.get_rownum()];
    #pragma omp parallel for
    for(index_t i = 0 ; i < B.get_rownum() ; i ++)
        for(index_t j = 0 ; j < B.get_colnum() ; j ++)
            B_TRANS[j*B.get_rownum()+i] = B_dense_data[i*B.get_colnum()+j];
    
    #pragma omp parallel for
    for(index_t i = 0 ; i < A.get_rownum() ; i ++)
        for(index_t j = 0 ; j < B.get_colnum() ; j ++){
            T3 tmp_rst = T3(0);
            for(index_t k = 0 ; k < A.get_colnum() ; k ++)
                tmp_rst += A_dense_data[i*A.get_colnum()+k]*B_TRANS[j*A.get_colnum()+k];
            rst_arr[i*B.get_colnum()+j] = tmp_rst;
        }
    rst.sparse_tag = false;
    rst.row_num = A.get_rownum();
    rst.col_num = B.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T3[A.get_rownum()*B.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*B.get_colnum()*sizeof(T3));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;
    if(B.is_sparse())
        delete[] B_dense_data;

    delete[] B_TRANS;
    return OP_SUCCESS;

}

/*
    Specified override to double-type matrix
    using AVX2 to boost calculation
*/
exception_t MMULT(const matrix<float>& A,const matrix<float>& B,matrix<float>& rst){
    if(A.get_colnum() != B.get_rownum()) return MATRIX_SIZE_INVALID;

    float* rst_arr = new float[A.get_rownum()*B.get_colnum()];
    float* A_dense_data = NULL;
    float* B_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    if(B.is_sparse())
        get_sparse_data(B,B_dense_data);
    else B_dense_data = B.dense_data;


    __m256 G1 = _mm256_setzero_ps();
    __m256 G2 = _mm256_setzero_ps();
    __m256 G3 = _mm256_setzero_ps();
    __m256 G4 = _mm256_setzero_ps();
    __m256 G_M = _mm256_setzero_ps();
    __m256 G1_DATA = _mm256_setzero_ps();
    __m256 G2_DATA = _mm256_setzero_ps();
    __m256 G3_DATA = _mm256_setzero_ps();
    __m256 G4_DATA = _mm256_setzero_ps();
    #pragma omp parallel for firstprivate(G1,G2,G3,G4,G_M,G1_DATA,G2_DATA,G3_DATA,G4_DATA)
    for(index_t row = 0 ; row < A.get_rownum() ; row ++){
        for(index_t k = 0 ; k + 31 < B.get_colnum() ; k += 32){
            G1 = _mm256_setzero_ps();
            G2 = _mm256_setzero_ps();
            G3 = _mm256_setzero_ps();
            G4 = _mm256_setzero_ps();
            for(index_t col = 0 ; col < A.get_colnum() ; col ++){
                G1_DATA = _mm256_loadu_ps(B_dense_data+col*B.get_colnum()+k);
                G2_DATA = _mm256_loadu_ps(B_dense_data+col*B.get_colnum()+k+8);
                G3_DATA = _mm256_loadu_ps(B_dense_data+col*B.get_colnum()+k+16);
                G4_DATA = _mm256_loadu_ps(B_dense_data+col*B.get_colnum()+k+24);
                G_M = _mm256_set1_ps(A_dense_data[row*A.get_colnum() + col]);
                G1 = _mm256_fmadd_ps(G1_DATA,G_M,G1);
                G2 = _mm256_fmadd_ps(G2_DATA,G_M,G2);
                G3 = _mm256_fmadd_ps(G3_DATA,G_M,G3);
                G4 = _mm256_fmadd_ps(G4_DATA,G_M,G4);
            }
            _mm256_storeu_ps(rst_arr+row*B.get_colnum()+k,G1);
            _mm256_storeu_ps(rst_arr+row*B.get_colnum()+k+8,G2);
            _mm256_storeu_ps(rst_arr+row*B.get_colnum()+k+16,G3);
            _mm256_storeu_ps(rst_arr+row*B.get_colnum()+k+24,G4);
        }
    }
    if(rst.dense_data != NULL){
        delete[] rst.dense_data;
        rst.dense_data = NULL;
    }

    if(B.get_colnum()%32 != 0){
        #pragma omp parallel for
        for(index_t row = 0 ; row < A.get_rownum() ; row ++)
            for(index_t col = B.get_colnum()-(B.get_colnum()%32) ; col < B.get_colnum() ; col ++){
                float result = 0;
                for(index_t k = 0 ; k < B.get_rownum() ; k ++)
                    result += A_dense_data[row*A.get_colnum()+k]*B_dense_data[k*B.get_colnum()+col];
                rst_arr[row*B.get_colnum()+col] = result;
            }
    }
    rst.sparse_tag = false;
    rst.row_num = A.get_rownum();
    rst.col_num = B.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new float[A.get_rownum()*B.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*B.get_colnum()*sizeof(float));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;
    if(B.is_sparse())
        delete[] B_dense_data;
    return OP_SUCCESS;
}


/*
    Dot product for matrix
    return exception and store the result in `rst`
*/
template<
    typename T1,
    typename T2,
    typename T3
>
exception_t DMULT(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst){
    if(A.get_rownum() != B.get_rownum() || A.get_colnum() != B.get_colnum())
        return MATRIX_SIZE_INVALID;

    T3* rst_arr = new T3[A.get_rownum()*A.get_colnum()];
    T3* A_dense_data = NULL;
    T3* B_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    if(B.is_sparse())
        get_sparse_data(B,B_dense_data);
    else B_dense_data = B.dense_data;

    #pragma omp parallel for
    for(index_t i = 0 ; i < A.row_num ; i ++)
        for(index_t j = 0 ; j < A.col_num ; j ++)
            rst_arr[i*A.col_num+j] = A_dense_data[i*A.col_num+j] * B_dense_data[i*A.col_num+j];
    rst.sparse_tag = false;
    rst.row_num = A.get_rownum();
    rst.col_num = A.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T3[A.get_rownum()*A.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*A.get_colnum()*sizeof(T3));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;
    if(B.is_sparse())
        delete[] B_dense_data;

    return OP_SUCCESS;    
}

/*
    Cross-type Matrix operation
    the user him/herself shoud check the validity of the 
    arithmetic operation and type conversion
    its not the designer's work to ensure that
    Matix-Matrix(Vector-Vector) addition
    return exception and store the result in `rst`
*/
template<
    typename T1,
    typename T2,
    typename T3
>
exception_t MADD(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst){
    if(A.get_rownum() != B.get_rownum() || A.get_colnum() != B.get_colnum())
        return MATRIX_SIZE_INVALID;

    T3* rst_arr = new T3[A.get_rownum()*A.get_colnum()];
    T3* A_dense_data = NULL;
    T3* B_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    if(B.is_sparse())
        get_sparse_data(B,B_dense_data);
    else B_dense_data = B.dense_data;

    #pragma omp parallel for
    for(index_t i = 0 ; i < A.row_num ; i ++)
        for(index_t j = 0 ; j < A.col_num ; j ++)
            rst_arr[i*A.col_num+j] = A_dense_data[i*A.col_num+j] + B_dense_data[i*A.col_num+j];
    rst.sparse_tag = false;        
    rst.row_num = A.get_rownum();
    rst.col_num = A.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T3[A.get_rownum()*A.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*A.get_colnum()*sizeof(T3));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;
    if(B.is_sparse())
        delete[] B_dense_data;

    return OP_SUCCESS;    
}

/*
    Cross-type Matrix operation
    the user him/herself shoud check the validity of the 
    arithmetic operation and type conversion
    its not the designer's work to ensure that
    Matix-Matrix(Vector-Vector) subtraction
    return exception and store the result in `rst`
*/
template<
    typename T1,
    typename T2,
    typename T3
>
exception_t MSUB(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst){
    if(A.get_rownum() != B.get_rownum() || A.get_colnum() != B.get_colnum())
        return MATRIX_SIZE_INVALID;

    T3* rst_arr = new T3[A.get_rownum()*A.get_colnum()];
    T3* A_dense_data = NULL;
    T3* B_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    if(B.is_sparse())
        get_sparse_data(B,B_dense_data);
    else B_dense_data = B.dense_data;

    #pragma omp parallel for
    for(index_t i = 0 ; i < A.row_num ; i ++)
        for(index_t j = 0 ; j < A.col_num ; j ++)
            rst_arr[i*A.col_num+j] = A_dense_data[i*A.col_num+j] - B_dense_data[i*A.col_num+j];
    rst.sparse_tag = false;        
    rst.row_num = A.get_rownum();
    rst.col_num = A.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T3[A.get_rownum()*A.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*A.get_colnum()*sizeof(T3));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;
    if(B.is_sparse())
        delete[] B_dense_data;

    return OP_SUCCESS;   
}

/*
    scalar-Matrix(scalar-Vector) multiplication
    return exception and store the result in `rst`
*/
template<typename T>
exception_t SMULT(const matrix<T>& A,T S,matrix<T>& rst){

    T* rst_arr = new T[A.get_rownum()*A.get_colnum()];
    T* A_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    #pragma omp parallel for
    for(index_t i = 0 ; i < A.row_num ; i ++)
        for(index_t j = 0 ; j < A.col_num ; j ++)
            rst_arr[i*A.col_num+j] = A_dense_data[i*A.col_num+j]*S;
    rst.sparse_tag = false;        
    rst.row_num = A.get_rownum();
    rst.col_num = A.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T[A.get_rownum()*A.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*A.get_colnum()*sizeof(T));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;

    return OP_SUCCESS;   
}

/*
    Matrix/Scalar(Vector/Scalar) division
    element A[i][j] will be replaced by A[i][j]/S
    return exception and store the result in `rst`
*/
template<typename T>
exception_t SDIV(const matrix<T>& A,double S,matrix<T>& rst){
    
    T* rst_arr = new T[A.get_rownum()*A.get_colnum()];
    T* A_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    #pragma omp parallel for
    for(index_t i = 0 ; i < A.row_num ; i ++)
        for(index_t j = 0 ; j < A.col_num ; j ++)
            rst_arr[i*A.col_num+j] = A_dense_data[i*A.col_num+j]/S;
    rst.sparse_tag = false;        
    rst.row_num = A.get_rownum();
    rst.col_num = A.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T[A.get_rownum()*A.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*A.get_colnum()*sizeof(T));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;

    return OP_SUCCESS;   
}

/*
    Reversed Scalar/Matrix(Scalar/Vector) division
    element A[i][j] will be replaced by S/A[i][j]
    return exception and store the result in `rst`
*/
template<typename T>
exception_t SDIV(double S,const matrix<T>& A,matrix<T>& rst){
    
    T* rst_arr = new T[A.get_rownum()*A.get_colnum()];
    T* A_dense_data = NULL;

    if(A.is_sparse())
        get_sparse_data(A,A_dense_data);
    else A_dense_data = A.dense_data;

    #pragma omp parallel for
    for(index_t i = 0 ; i < A.row_num ; i ++)
        for(index_t j = 0 ; j < A.col_num ; j ++)
            rst_arr[i*A.col_num+j] = S/A_dense_data[i*A.col_num+j];
    rst.sparse_tag = false;        
    rst.row_num = A.get_rownum();
    rst.col_num = A.get_colnum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    rst.dense_data = new T[A.get_rownum()*A.get_colnum()];
    memcpy(rst.dense_data,rst_arr,A.get_rownum()*A.get_colnum()*sizeof(T));
    delete[] rst_arr;
    
    if(A.is_sparse())
        delete[] A_dense_data;

    return OP_SUCCESS;   
}

/*
    Vector cross mult with result stored in `rst` 
    return possible exception
*/
template<typename T>
exception_t VCMULT3D(const matrix<T>& A,const matrix<T>& B,matrix<T>& rst){
    if(
        !(
            (A.get_rownum() == 1 && A.get_colnum() == 3) ||
            (A.get_rownum() == 3 && A.get_colnum() == 1) ||
            (B.get_rownum() == 1 && B.get_colnum() == 3) ||
            (B.get_rownum() == 3 && B.get_colnum() == 1)
        )
    ) return MATRIX_SIZE_INVALID;
    T x1,x2,x3,y1,y2,y3,z1,z2,z3;
    x1 = A.get_val(0,0);
    x2 = B.get_val(0,0);
    y1 = (A.get_rownum() == 1 && A.get_colnum() == 3) ? A.get_val(0,1) : A.get_val(1,0);
    y2 = (B.get_rownum() == 1 && B.get_colnum() == 3) ? B.get_val(0,1) : B.get_val(1,0);
    z1 = (A.get_rownum() == 1 && A.get_colnum() == 3) ? A.get_val(0,2) : A.get_val(2,0);
    z2 = (B.get_rownum() == 1 && B.get_colnum() == 3) ? B.get_val(0,2) : B.get_val(2,0);

    rst.resize(3,1);
    rst.set_val(0,0,y1*z2-y2*z1);
    rst.set_val(1,0,x1*z2-x2*z1);
    rst.set_val(2,0,x1*y2-x2*y1);
}

/*
    Matrix transpose
    return exception and store result in `rst`
    remenber to handle with the situation that 
    `A` and `rst` points to the same address
*/
template<typename T>
exception_t MTRANS(const matrix<T>& A,matrix<T>& rst){
    if(&A == &rst){
        exception_handler::puts_err("can't use MTRANS(A,RST) while A and RST is the same matrix");
        return MATRIX_DATA_INVALID;
    }
    if(A.is_sparse()){
        rst.sparse_tag = true;
        rst.sparse_data.clear();
        rst.row_num = A.col_num;
        rst.col_num = A.row_num;
        for(triplet<T> t : A.sparse_data)
            rst.sparse_data.emplace_back(t.getcol(),t.getrow(),t.getdata());
        return OP_SUCCESS;
    }

    T* rst_arr = new T[A.get_rownum()*A.get_colnum()];
    T* A_dense_data = A.dense_data;


    #pragma omp parallel for
    for(index_t i = 0 ; i < A.get_rownum() ; i ++)
        for(index_t j = 0 ; j < A.get_colnum() ; j ++)
            rst_arr[j*A.get_rownum()+i] = A_dense_data[i*A.get_colnum()+j];

    

    rst.sparse_tag = false;
    rst.row_num = A.get_colnum();
    rst.col_num = A.get_rownum();
    if(rst.dense_data != NULL)
        delete[] rst.dense_data;
    
    
    rst.dense_data = new T[A.get_rownum()*A.get_colnum()];

    for(index_t i = 0 ; i < rst.row_num ; i ++)
        for(index_t j = 0 ; j < rst.col_num ; j ++)
            rst.dense_data[i*rst.col_num+j] = rst_arr[i*rst.col_num+j];
    
    delete[] rst_arr;
    return OP_SUCCESS;
}

/*
    Matrix transpose
    transpose the given matrix A
    directly operate on A
*/
template<typename T>
exception_t MTRANS(matrix<T>& A){
    matrix<T> tmp;
    exception_t ans = MTRANS(A,tmp);
    A = matrix<T>(tmp);
    return ans;
}

/*
    Matrix conjugation calculate
    only support the matrix with data type complex<T>
*/
template<typename T>
exception_t MCONJ(const matrix<std::complex<T>>& A,matrix<std::complex<T>>& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;

    MTRANS(A,rst);

    for(int i = 0 ; i < A.get_rownum() ; i ++)
        for(int j = 0 ; j < A.get_colnum() ; j ++){
            std::complex<T> val = rst.get_val(i,j);
            std::complex<T> nval(val.real(),val.imag()*(-1));
            rst.set_val(i,j,nval);
        }
    return OP_SUCCESS;
}

/*
    Matrix conjugation calculate
    only support the matrix with data type complex<T>
    directly operate on A
*/
template<typename T>
exception_t MCONJ(matrix<std::complex<T>>& A){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    

    MTRANS(A);
    
    for(int i = 0 ; i < A.get_rownum() ; i ++)
        for(int j = 0 ; j < A.get_colnum() ; j ++){
            std::complex<T> val = A.get_val(i,j);
            std::complex<T> nval(val.real(),val.imag()*(-1));
            A.set_val(i,j,nval);
        }
    return OP_SUCCESS;
}



/*---------------- Matrix vector reduction operations ----------------*/



const tag_t EXT_MIN_TAG=0x10000001;
const tag_t EXT_MAX_TAG=0x10000002;

/*
    by the given OP_TAG(EXT_XXX_TAG)
    find the max or min value in a specific row from matrix
*/
template<typename T>
exception_t MEXTR_ROW(const matrix<T>& A,index_t row_num,tag_t OP_TAG,T& rst){
     if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
     if(row_num<0 || row_num>A.get_rownum())return INDEX_OUT_OF_BOUND;
     if(!A.is_valid())return MATRIX_DATA_INVALID;
     rst=A.get_val(row_num,0);

    if(OP_TAG==EXT_MIN_TAG)
    {
        for(int i=1; i<A.get_colnum(); i++)
        {
            if(rst>A.get_val(row_num,i))rst=A.get_val(row_num,i);
        }


    }
    else if(OP_TAG==EXT_MAX_TAG)
    {
        for(int i=1; i<A.get_colnum(); i++)
        {
            if(rst<A.get_val(row_num,i))rst=A.get_val(row_num,i);
        }

    }
     return OP_SUCCESS;
}

/*
    by the given OP_TAG(EXT_XXX_TAG)
    find the max or min value in a specific col from matrix
*/
template<typename T>
exception_t MEXTR_COL(const matrix<T>& A,index_t col_num,tag_t OP_TAG,T& rst){
     if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
     if(col_num<0 || col_num>A.get_colnum())return INDEX_OUT_OF_BOUND;
     if(!A.is_valid())return MATRIX_DATA_INVALID;
     rst=A.get_val(0,col_num);

    if(OP_TAG==EXT_MIN_TAG)
    {
        for(int i=1; i<A.get_rownum(); i++)
        {
            if(rst>A.get_val(i,col_num))rst=A.get_val(i,col_num);
        }


    }
    else if(OP_TAG==EXT_MAX_TAG)
    {
        for(int i=1; i<A.get_rownum(); i++)
        {
             if(rst<A.get_val(i,col_num))rst=A.get_val(i,col_num);
        }

    }
     return OP_SUCCESS;
}

/*
    by the given OP_TAG(EXT_XXX_TAG)
    find the max or min value in the whole matrix
*/
template<typename T>
exception_t MEXTR(const matrix<T>& A,tag_t OP_TAG,T& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;

     rst=A.get_val(0,0);
    if(OP_TAG==EXT_MIN_TAG)
    {
        for(int i=0; i<A.get_rownum(); i++)
        {
            for(int j=0; j<A.get_colnum(); j++)
            {
                if(rst>A.get_val(i,j))rst=A.get_val(i,j);
            }
        }

    }
    else if(OP_TAG==EXT_MAX_TAG)
    {
       for(int i=0; i<A.get_rownum(); i++)
        {
            for(int j=0; j<A.get_colnum(); j++)
            {
                if(rst<A.get_val(i,j))rst=A.get_val(i,j);
            }
        }

    }
     return OP_SUCCESS;
}

/*
    find the sum in a specific row from matrix
*/
template<typename T>
exception_t MSUM_ROW(const matrix<T>& A,index_t row_num,T& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    if(row_num<0 || row_num>A.get_rownum())return INDEX_OUT_OF_BOUND;

    rst=A.get_val(row_num,0);
    for(int i=1; i<A.get_colnum(); i++)
    {
       rst+=A.get_val(row_num,i);
    }
     return OP_SUCCESS;
}

/*
    find the sum in a specific col from matrix
*/
template<typename T>
exception_t MSUM_COL(const matrix<T>& A,index_t col_num,T& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    if(col_num<0 || col_num>A.get_colnum())return INDEX_OUT_OF_BOUND;

    rst=A.get_val(0,col_num);
    for(int i=1; i<A.get_rownum(); i++)
    {
       rst+=A.get_val(i,col_num);
    }
     return OP_SUCCESS;
}


/*
    find the sum of the whole matrix --by WTA
*/
template<typename T>
exception_t MSUM(const matrix<T>& A,T& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    rst=0;
    for(int i=0; i<A.get_rownum(); i++)
    {
        for(int j=0; j<A.get_colnum(); j++)
        {
            rst+=A.get_val(i,j);
        }
    }
     return OP_SUCCESS;
}

/*
    find the average value in a specific row from matrix --by WTA
*/
template<typename T>
exception_t MAVG_ROW(const matrix<T>& A,index_t row_num,T& rst){
    T sum;
    exception_t e=MSUM_ROW(A,row_num,sum);
    if (e!= OP_SUCCESS){
        return e;
    }
    if(row_num>=A.get_rownum()||row_num<0){
        return INDEX_OUT_OF_BOUND;
    }
    rst=sum/A.get_colnum();
    return OP_SUCCESS;
}

/*
    find the average value in a specific col from matrix --by WTA
*/
template<typename T>
exception_t MAVG_COL(const matrix<T>& A,index_t col_num,T& rst){
    T sum;
    exception_t e=MSUM_COL(A,col_num,sum);
    if (e!= OP_SUCCESS){
        return e;
    }
    if(col_num>=A.get_colnum()||col_num<0){
        return INDEX_OUT_OF_BOUND;
    }
    rst=sum/A.get_rownum();
    return OP_SUCCESS;
}

/*
    find the average value of the whole matrix --by WTA
*/
template<typename T>
exception_t MAVG(const matrix<T>& A,T& rst){
    T sum;
    exception_t e=MSUM(A,sum);
    if (e!= OP_SUCCESS){
        return e;
    }
    rst=sum/(A.get_rownum()*A.get_colnum());
    return OP_SUCCESS;
}


/*
    find the trace of matrix A and store it to rst
*/
template<typename T>
exception_t MTRACE(const matrix<T>& A,T& rst){
    if(A.get_colnum() != A.get_rownum()) return MATRIX_SIZE_INVALID;
    rst = 0;
    for(index_t i = 0 ; i < A.get_colnum() ; i ++)
        rst += A.get_val(i,i);
}

/*
    find the eigenvalue and the eigenvector of matrix A and store it to rstVector and rstValue --by WTA
*/
template<typename T>
exception_t MEGVLCT(const matrix<T>& A,matrix<T>& rstValue,matrix<T>& rstVector,double EPS,long MAX_Count){
    if(A.get_colnum()!=A.get_rownum()||rstVector.get_colnum()!=rstVector.get_rownum()
    ||rstVector.get_rownum()!=A.get_rownum()||rstValue.get_colnum()!=1||rstValue.get_rownum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    int M=A.get_rownum();
    matrix<T> ACOPY = A;
    //initialize the eigenvector matrix
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if(i==j) rstVector.set_val(i,j,1);
            else rstVector.set_val(i,j,0);
        }
    }
    
    long cnt=0;

    while (1)
    {   
        //if cnt>=MAX_Count, then finish the calculation
        if(cnt >= MAX_Count){
            break;
        }
        cnt++;
        //find the max value in the matrix that not in major line
        T max_value=ACOPY.get_val(0,1);
        int nRow=0,nCol=1;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < M; j++){
                T abs=fabs(ACOPY.get_val(i,j));
                if((i!=j) && (abs > max_value)){
                    max_value=abs;
                    nRow=i;
                    nCol=j;
                }
            }
        }
        //if the max value is less than EPS, then break
        if(max_value < EPS){
            break;
        }

        T pp=ACOPY.get_val(nRow,nRow);
        T qq=ACOPY.get_val(nCol,nCol);
        T pq=ACOPY.get_val(nRow,nCol);
        
        T theta=atan2(2*pq,pp-qq)*0.5;
        T c=cos(theta);
        T s=sin(theta);
        T s2=sin(2*theta);
        T c2=cos(2*theta);

        ACOPY.set_val(nRow,nRow,c*c*pp+2*s*c*pq+s*s*qq);
        ACOPY.set_val(nCol,nCol,c*c*qq-2*s*c*pq+s*s*pp);
        ACOPY.set_val(nRow,nCol,c2*pq+s2*(qq-pp)*0.5);
        ACOPY.set_val(nCol,nRow,c2*pq+s2*(qq-pp)*0.5);

        for (int i = 0; i < M; i++)
        {
            if((i!=nRow) && (i!=nCol)){
                T tempr=ACOPY.get_val(i,nRow);
                T tempc=ACOPY.get_val(i,nCol);
                ACOPY.set_val(i,nRow,c*tempr+s*tempc);
                ACOPY.set_val(i,nCol,-s*tempr+c*tempc);
            }
        }

        for (int j = 0; j < M; j++){
            if((j!=nRow) && (j!=nCol)){
                T tempr=ACOPY.get_val(nRow,j);
                T tempc=ACOPY.get_val(nCol,j);
                ACOPY.set_val(nRow,j,c*tempr+s*tempc);
                ACOPY.set_val(nCol,j,-s*tempr+c*tempc);
            }
        }
        
        for(int i =0; i < M ; i++){
            T tempr=rstVector.get_val(i,nRow);
            T tempc=rstVector.get_val(i,nCol);
            rstVector.set_val(i,nRow,c*tempr+s*tempc);
            rstVector.set_val(i,nCol,-s*tempr+c*tempc);
        }
    }

    for (int i = 0; i < M; i++)
    {
        rstValue.set_val(i,0,ACOPY.get_val(i,i));
    }

    return OP_SUCCESS;
}

/*
    find the convolutional of two matrices store result in `rst`
    with convolutional core `CONV_CORE` and Pending matrix A
*/
template<typename T>
exception_t MCONV(const matrix<T>& CONV_CORE,const matrix<T>& A,matrix<T>& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(CONV_CORE.get_colnum()!=CONV_CORE.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    if(!CONV_CORE.is_valid())return MATRIX_DATA_INVALID;
    if(CONV_CORE.get_colnum()>A.get_colnum())return INDEX_OUT_OF_BOUND;

    rst=matrix<T>(A.get_rownum()-CONV_CORE.get_rownum()+1, A.get_colnum()-CONV_CORE.get_colnum()+1);

    for (int i = 0; i < rst.get_rownum(); ++i)
	{
		for (int j = 0; j < rst.get_colnum(); ++j)
		{
			T sum = 0;
			for (int m = 0; m < CONV_CORE.get_rownum(); ++m)
			{
				for (int n = 0; n < CONV_CORE.get_colnum(); ++n)
				{
					sum +=A.get_val(i + m,j + n)* CONV_CORE.get_val(m,n);
				}
			}
			rst.set_val(i,j,sum);

		}
	}

      return OP_SUCCESS;
}

/*
    calculate the determinant of a matrix
*/
template<typename T>
T det(const matrix<T>& A,int size)
{
    T rst=0;
    if(A.get_colnum()==1){
        rst=A.get_val(0,0);
    }
    else if(A.get_colnum()==2)
    {
        rst=A.get_val(0,0)*A.get_val(1,1)-A.get_val(1,0)*A.get_val(0,1);
    }
    else if(A.get_colnum()>=3)
    {
        for (msize_t i = 0; i < A.get_colnum(); ++i)
        {
            matrix<T> m=matrix<T>(A.get_colnum()-1,A.get_colnum()-1);

            for(int p=0; p<A.get_colnum()-1; p++)
            {
                for(int q=0; q<A.get_colnum()-1; q++)
                {
                    if(q<i)
                        m.set_val(p,q,A.get_val(p+1,q));
                    else
                        m.set_val(p,q,A.get_val(p+1,q+1));
                }
            }

            if(i%2==0)
                rst+= A.get_val(0,i)*det(m,size-1);
            else if(i%2==1)
                rst-= A.get_val(0,i)*det(m,size-1);

        }

    }
    return rst;
}

/*
    calculate the determinant of a matrix
*/
template<typename T>
exception_t MDET(const matrix<T>& A,T& rst){
    if(A.get_colnum()!=A.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;

    rst=0;

    rst=det(A,A.get_colnum());

    return OP_SUCCESS;
}

/*
    calculate the algebraic complement of the matrix --by WTA
*/

template<typename T>
exception_t MALCO(const matrix<T>& A,matrix<T>& complement,int i,int j){
    if(A.get_colnum()!=A.get_rownum()||A.get_colnum()!=complement.get_colnum()+1||complement.get_colnum()!=complement.get_rownum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid()) return MATRIX_DATA_INVALID;
    for(int p=0; p<A.get_rownum()-1; p++)
    {
        for(int q=0; q<A.get_colnum()-1; q++)
        {
            complement.set_val(p,q,A.get_val(p+(p>=i),q+(q>=j)));
        }
    }
    return OP_SUCCESS;
}

/*
    calculate the adjoint matrix of the matrix --by WTA
*/
template<typename T>
exception_t MADJ(const matrix<T>& A,matrix<T>& adj,int size){
    if(A.get_colnum()!=A.get_rownum()||A.get_colnum()!=size||adj.get_colnum()!=adj.get_rownum()||adj.get_colnum()!=A.get_colnum())return MATRIX_SIZE_INVALID;
    if(!A.is_valid())return MATRIX_DATA_INVALID;
    for (msize_t i = 0; i < adj.get_rownum(); i++)
    {
        for (msize_t j = 0; j < adj.get_colnum(); j++)
        {
            matrix<T> m=matrix<T>(A.get_colnum()-1,A.get_colnum()-1);

            exception_t e=MALCO(A,m,i,j);
            if(e!=OP_SUCCESS)return e;

            if((i+j)%2==0){
                adj.set_val(j,i,det(m,size-1));
            }
            else if((i+j)%2==1){
                adj.set_val(j,i,-det(m,size-1));
            }
        }
    }
    return OP_SUCCESS;
}

/*
    calculate the inverse of a matrix --by WTA
*/
template<typename T>
exception_t MINVER(const matrix<T>& A,matrix<T>& rst){
    if(A.get_colnum() != A.get_rownum()){
        return MATRIX_NOT_INVERTIBLE;
    }
    if(!A.is_valid()){
        return MATRIX_DATA_INVALID;
    }
    T det=0;
    exception_t e=MDET(A,det);    //calculate the determinant    ?????
    if(e!=OP_SUCCESS){
        return e;
    }
    if(det==0){
        return MATRIX_NOT_DETERMINANT;
    }
    matrix<T> adj=matrix<T>(A.get_colnum(),A.get_colnum());
    exception_t e0 = MADJ(A,adj,A.get_colnum());
    if(e0!=OP_SUCCESS){
        return e0;
    }
    exception_t e1=SMULT(adj,1/det,rst);
    if(e1!=OP_SUCCESS){
        return e1;
    }
    return OP_SUCCESS;
}

