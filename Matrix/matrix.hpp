#pragma once
#include"exception_handler.hpp"
#include"triplet.hpp"
#include"exceptions.hpp"
#include<cstdint>
#include<vector>
#include<cstdio>
#include<iostream>
#include<cstring>
#include<omp.h>
#include<opencv2/opencv.hpp>
//unsigned long long group
typedef uint64_t tag_t;
typedef uint64_t msize_t;
typedef uint64_t exception_t;

//long long group
typedef int64_t index_t;
template<typename value_t>
class matrix{
private:
    /*
        Two unsigned long long type variable which records the row and column num. 
    */
    msize_t row_num;
    msize_t col_num;
    
    /*
        If the matrix is sparse, we use 'triplet' vector to store,
        else we use 'dense_data' array to store.
        Note: All data should store in 1-dimension. 
        dense_data will be and only be NULL if the matrix is defined to be sparse
    */
    value_t* dense_data;
    std::vector<triplet<value_t>> sparse_data;

    /*
        is_sparse is used to sign whether matrix is sparse or not.
        true: the matrix is sparse.
        false: the matrix is not sparse.
    */
    bool sparse_tag;

    friend exception_t MMULT(const matrix<float>& A,const matrix<float>& B,matrix<float>& rst);

    template<
        typename T1,
        typename T2,
        typename T3
    >
    friend exception_t MMULT(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst);

    template<
        typename T1,
        typename T2,
        typename T3
    >
    friend exception_t DMULT(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst);

    template<
        typename T1,
        typename T2,
        typename T3
    >
    friend exception_t MADD(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst);

    template<
        typename T1,
        typename T2,
        typename T3
    >
    friend exception_t MSUB(const matrix<T1>& A,const matrix<T2>& B,matrix<T3>& rst);

    template<typename T>
    friend exception_t SMULT(const matrix<T>& A,T S,matrix<T>& rst);

    template<typename T>
    friend exception_t SDIV(const matrix<T>& A,double S,matrix<T>& rst);

    template<typename T>
    friend exception_t SDIV(double S,const matrix<T>& A,matrix<T>& rst);

    template<typename T>
    friend exception_t MTRANS(const matrix<T>& A,matrix<T>& rst);

public:
    /*
        get the num of col of the matrix
    */
    msize_t get_colnum() const{
        return col_num;
    }

    /*
        get the num of row of the matrix
    */
    msize_t get_rownum() const{
        return row_num;
    }

    /*
        get the data address (unchangable)
    */
    const value_t* get_dense_data() const{
        return dense_data;
    }
    
    /*
        get the sparse data
    */
    const std::vector<triplet<value_t>>& get_sparse_data() const{
        return sparse_data;
    }
    /*
        The default constructor , init the matrix data to a single 0
        its freaking hard to define a default matrix for an unknown type
    */
    matrix(){
        dense_data = new value_t[1];
        dense_data[0] = value_t(0);
        row_num = 1;
        col_num = 1;
        sparse_tag = false;
    }

    /*
        This is a init constructor to fill all elements to zero.
    */
    matrix(index_t row_num, index_t col_num){
        if(row_num <= 0 || col_num <= 0){
            exception_handler::puts_err("exception: invalid row_num or col_num, calling default constructor");
            matrix();
        }else{
            dense_data = new value_t[row_num*col_num];
            this->row_num = row_num;
            this->col_num = col_num;
            this->sparse_tag = false;
            #pragma omp parallel for num_threads(2)
            for(index_t i = 0 ; i < row_num ; i ++)
                for(index_t j = 0 ; j < col_num ; j ++)
                    dense_data[i*col_num+j] = value_t(0);
        }
    }

    
    static const tag_t SPARSE_TAG = 0x10000001;
    static const tag_t DENSE_TAG = 0x10000002;
    /*
        This is a standadized constructor which we expect the user to choose.
        The last parameter we expect the user to choose between matrix::SPARSE_TAG and matrix::DENSE_TAG 
    */
    matrix(const value_t* data, index_t row_num, index_t col_num,const tag_t MAT_TAG){
        if(row_num <= 0 || col_num <= 0){
            exception_handler::puts_err("exception: invalid row_num or col_num, calling default constructor");
            matrix();
        }else{
            this->row_num = row_num;
            this->col_num = col_num;
            if(MAT_TAG == DENSE_TAG){        
                dense_data = new value_t[row_num*col_num];
                sparse_tag = false;
                #pragma omp parallel for num_threads(2)
                for(index_t i = 0 ; i < row_num ; i ++)
                    for(index_t j = 0 ; j < col_num ; j ++)
                        dense_data[i*col_num+j] = data[i*col_num+j];
                
            }
            if(MAT_TAG == SPARSE_TAG){
                dense_data = NULL;
                sparse_tag = true;
                for(index_t i = 0 ; i < row_num ; i ++)
                    for(index_t j = 0 ; j < col_num ; j ++)
                        if(data[i*col_num+j] != value_t(0))
                            sparse_data.push_back(triplet<value_t>(i,j,data[i*col_num+j]));      
            }
        }
    }

    /*
        default cpy constructor
    */
    matrix(const matrix<value_t>& A){
        this->row_num = A.get_rownum();
        this->col_num = A.get_colnum();
        sparse_tag = A.is_sparse();
        if(!A.is_sparse()){
            this->dense_data = new value_t[row_num*col_num];
            memcpy(dense_data,A.dense_data,row_num*col_num*sizeof(value_t));
        }else{
            dense_data = NULL;
            sparse_data.clear();
            for(index_t i = 0 ; i < A.sparse_data.size() ; i ++)
                sparse_data.push_back(A.sparse_data[i]);
        }
    }

    /*
        OPENCV constructor
    */
#ifdef OPENCV_ALL_HPP
    matrix(const cv::Mat& CVMat){
        this->sparse_tag = false;
        this->row_num = CVMat.rows;
        this->col_num = CVMat.cols;
        this->dense_data = new value_t[row_num*col_num];
        for(int i = 0 ; i < CVMat.rows ; i ++)
            for(int j = 0 ; j < CVMat.cols ; j ++)
                this->dense_data[i*col_num+j] = CVMat.at<double>(i,j);
    }
#endif
    /*
        temporarily disabled constructor for supporting OpenCV
    */
    //  matrix(<Some_OpenCV_Type> CVMatrix);

    /*
        desctructor, remember to delete the manually allocated space in any place
    */
    ~matrix(){
        if(dense_data != NULL)
            delete[] dense_data;
    }

    /*
        Function used to judge sparse or not
    */
    bool is_sparse() const{
        return sparse_tag;
    };

    /*
        Function to check whether the data is NULL or not
    */
    bool is_valid() const{
        if(is_sparse())
            return sparse_data.size() != 0;
        else return dense_data != NULL;
    };

    /*
        clean all data in the matrix
    */
    void clear(){
        col_num = 1;
        row_num = 1;
        if(!is_sparse()){
            delete[] dense_data;
            dense_data = new value_t[1];
            dense_data[0] = value_t(0);
            return;
        }
        sparse_data.clear();
        
    }

    /*
        resize the matrix, retain the data as far as possible
        and fill 0 to the places with no data before
    */
    void resize(index_t row,index_t col){
        if(row <= 0 || col <= 0){
            exception_handler::puts_err("invalid index to resize, please check row and col");
            return;
        }
        if(!is_sparse()){
            value_t* new_data = new value_t[row*col];
            #pragma omp parallel for num_threads(2)
            for(index_t i = 0 ; i < row ; i ++)
                for(index_t j = 0 ; j < col ; j ++)
                    new_data[i*col+j] = value_t(0);
            if(dense_data == NULL){
                dense_data = new_data;
                return;
            }else{
                msize_t row_bound = std::min(row_num,msize_t(row));
                msize_t col_bound = std::min(col_num,msize_t(col));
                #pragma omp parallel for num_threads(2)
                for(index_t i = 0 ; i < row_bound ; i ++)
                    for(index_t j = 0 ; j < col_bound ; j ++)
                        new_data[i*col+j] = dense_data[i*col_num+j];
                delete[] dense_data;
                dense_data = new_data;
                row_num = row;
                col_num = col;
                return;
            }
        }else{
            row_num = row;
            col_num = col;
            std::vector<triplet<value_t>> back_up;
            for(index_t i = 0 ; i < sparse_data.size() ; i ++)
                back_up.push_back(sparse_data[i]);
            sparse_data.clear();
            for(index_t i = 0 ; i < back_up.size() ; i ++)
                if(back_up[i].getrow() < row && back_up[i].getcol() < col)
                    sparse_data.push_back(back_up[i]);
            return;
        }
    }

    /*
        resize the matrix, clean all data before and fill 0 to the new matrix
    */
    void resize_and_clear(index_t row,index_t col){
        if(row <= 0 || col <= 0){
            exception_handler::puts_err("invalid index to resize, please check row and col");
            return;
        }
        if(!is_sparse()){
            value_t* new_data = new value_t[row*col];
            #pragma omp parallel for num_threads(2)
            for(index_t i = 0 ; i < row ; i ++)
                for(index_t j = 0 ; j < col ; j ++)
                    new_data[i*col+j] = value_t(0);
            if(dense_data == NULL){
                dense_data = new_data;
                return;
            }else{
                delete[] dense_data;
                dense_data = new_data;
                row_num = row;
                col_num = col;
                return;
            }
        }else{
            row_num = row;
            col_num = col;
            sparse_data.clear();
            return;
        }
    }

    /*
        slice the matrix, only retain the data between s_row and e_row
    */
    void slice_row(index_t s_row,index_t e_row){
        if(s_row < 0 || e_row >= row_num || s_row > e_row){
            exception_handler::puts_err("exception: s_row or e_row invalid, nothing changed");
            return;
        }
        if(is_sparse()){
            std::vector<triplet<value_t>> bak_data;
            for(triplet<value_t> data : sparse_data)
                bak_data.push_back(data);
            sparse_data.clear();
            for(triplet<value_t> data : bak_data)
                if(data.getrow() >= s_row && data.getrow() <= e_row)
                    sparse_data.push_back(
                        triplet<value_t>(data.getrow()-s_row , data.getcol() , data.getdata())
                    );
            row_num = e_row - s_row + 1;
        }else{
            value_t* new_data = new value_t[(e_row-s_row+1)*col_num];
            #pragma omp parallel for num_threads(2)
            for(index_t i = s_row ; i <= e_row ; i ++)
                for(index_t j = 0 ; j < col_num ; j ++)
                    new_data[(i-s_row)*col_num+j] = dense_data[i*col_num+j];
            row_num = e_row - s_row + 1;
            delete[] dense_data;
            dense_data = new_data;
        }
    }

    /*
        slice the matrix, only retain the data between s_col and e_col
    */
    void slice_col(index_t s_col,index_t e_col){
        if(s_col < 0 || e_col >= col_num || s_col > e_col){
            exception_handler::puts_err("exception: s_col or e_col invalid, nothing changed");
            return;
        }
        if(is_sparse()){
            std::vector<triplet<value_t>> bak_data;
            for(triplet<value_t> data : sparse_data)
                bak_data.push_back(data);
            sparse_data.clear();
            for(triplet<value_t> data : bak_data)
                if(data.getcol() >= s_col && data.getcol() <= e_col)
                    sparse_data.push_back(
                        triplet<value_t>(data.getrow() , data.getcol()-s_col , data.getdata())
                    );
            col_num = e_col - s_col + 1;
        }else{
            value_t* new_data = new value_t[row_num*(e_col-s_col+1)];
            #pragma omp parallel for num_threads(2)
            for(index_t i = 0 ; i < row_num ; i ++)
                for(index_t j = s_col ; j <= e_col ; j ++)
                    new_data[i*(e_col-s_col+1)+j-s_col] = dense_data[i*col_num+j];
            col_num = e_col - s_col + 1;
            delete[] dense_data;
            dense_data = new_data;
        }
    }

    /*
        This function is used to get the VALUE of dense_data[i][j] or in triplet vector
        from the `dest`
        return exception whenever an excepetion occurs
    */
    value_t get_val(index_t i, index_t j) const{
        if(i < 0 || j < 0 || i >= row_num || j >= col_num){
            exception_handler::puts_err("exception: invalid index, please check i or j, returned value_t(0)");
            return value_t(0);
        }
        if(!is_sparse())
            return dense_data[i*col_num+j];
        else{
            for(index_t iterator = 0 ; iterator < sparse_data.size() ; iterator ++)
                if(sparse_data[iterator].getrow() == i && sparse_data[iterator].getcol() == j)
                    return sparse_data[iterator].getdata();
            return 0;
        }

        
    }
    
    /*
        This function is used to set the VALUE of dense_data[i][j] or in triplet vector
        from the `src`
        return exception whenever an excepetion occurs
    */
    void set_val(index_t i, index_t j,value_t src){
        if(i < 0 || j < 0 || i >= row_num || j >= col_num){
            exception_handler::puts_err("exception: invalid index of i or j, value unsetted");
            return;
        }
        if(!is_sparse())
            dense_data[i*col_num+j] = src;
        else{
            for(index_t iterator = 0 ; iterator < sparse_data.size() ; iterator ++)
                if(sparse_data[iterator].getrow() == i && sparse_data[iterator].getcol() == j){
                    sparse_data[iterator].setdata(src);
                    return;
                }
            sparse_data.push_back(triplet<value_t>(i,j,src));
        }
    }

    /*
        The only operator to override is `==`
        override to other arthmetic operators might be quite troublesome
        if any exception occurs, just return false
        remember to handle null == null or sth else
    */
    bool operator == (const matrix<value_t>& A) const{
        if(this == &A) return true;
        if(col_num != A.get_colnum() || row_num != A.get_rownum()) return false;
        const value_t* own_data = NULL;
        value_t* tmp_own = NULL;
        const value_t* comp_data = NULL;
        value_t* tmp_comp = NULL;
        auto extract_data = [&](
            value_t*& dest_data,
            const std::vector<triplet<value_t>>& src_data,
            index_t row,index_t col
        ) -> void{
            dest_data = new value_t[row*col];
            #pragma omp parallel for num_threads(2)
            for(index_t i = 0 ; i < row ; i ++)
                for(index_t j = 0 ; j < col ; j ++)
                    dest_data[i*col+j] = value_t(0);
            for(index_t iterator = 0 ; iterator < src_data.size() ; iterator++){
                index_t i = src_data[iterator].getrow();
                index_t j = src_data[iterator].getcol();
                value_t data = src_data[iterator].getdata();
                dest_data[i*col+j] = data;
            }
        };
        if(is_sparse()){
            extract_data(tmp_own,sparse_data,row_num,col_num);
            own_data = tmp_own;
        }else own_data = dense_data;
        if(A.is_sparse()){
            extract_data(tmp_comp,A.get_sparse_data(),row_num,col_num);
            comp_data = tmp_comp;
        }else comp_data = A.get_dense_data();
        bool neq_flag = false;
        #pragma omp parallel for
        for(index_t i = 0 ; i < row_num ; i ++)
            for(index_t j = 0 ; j < col_num ; j ++){
                if(neq_flag) break;
                if(own_data[i*col_num+j] != comp_data[i*col_num+j]){
                    neq_flag = true;
                    break;
                }
            }
        return !neq_flag;
        
    }
    
    void operator = (const matrix<value_t>& A){
        this->row_num = A.get_rownum();
        this->col_num = A.get_colnum();
        sparse_tag = A.is_sparse();
        if(!A.is_sparse()){
            this->dense_data = new value_t[row_num*col_num];
            memcpy(dense_data,A.dense_data,row_num*col_num*sizeof(value_t));
        }else{
            dense_data = NULL;
            sparse_data.clear();
            for(index_t i = 0 ; i < A.sparse_data.size() ; i ++)
                sparse_data.push_back(A.sparse_data[i]);
        }
        
    }

#ifdef OPENCV_ALL_HPP
    operator cv::Mat() const{
        cv::Mat M2R;
        float* fdata = new float[row_num*col_num];
        memset(fdata,0,sizeof(float)*col_num*row_num);
        if(this->is_sparse()){
            for(triplet<value_t> t : sparse_data)
                fdata[t.getrow()*col_num+t.getcol()] = (float)(t.getdata());
        }else{
            msize_t tsize = col_num*row_num;
            for(msize_t i = 0 ; i < tsize ; i ++)
                fdata[i] = (float)(dense_data[i]);
        }
        return cv::Mat(row_num,col_num,CV_32F,fdata);
    }
#endif

    friend std::ostream& operator << (std::ostream& os,const matrix<value_t>& A){
        
        for(index_t i = 0 ; i < A.get_rownum() ; i ++){
            for(index_t j = 0 ; j < A.get_colnum() ; j ++)
                os<<A.get_val(i,j)<<" ";
            os<<std::endl;
        }
        return os;
    }
};


