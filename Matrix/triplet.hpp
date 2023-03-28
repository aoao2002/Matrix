#pragma once
#include"exception_handler.hpp"

typedef int64_t index_t;

template<typename value_t>
class triplet{

private:
    index_t row,col;
    value_t data;

public:
    /*
        default constructor, init index into invalid value
    */
    triplet(){
        row = -1;
        col = -1;
        data = 0;
    }

    /*
        constructor with given three values
    */
    triplet(index_t row, index_t col, value_t data){
        if(row < 0 || col < 0){
            exception_handler::puts_err("Index out of bound exception, set row and col to -1");
            triplet();
        }else{
            this->row = row;
            this->col = col;
            this->data = data;
        }
    }

    /*
        swap i and j
    */
    void swap_idx(){
        std::swap(row,col);
    }

    /*
        check whether the index is valid or not
    */
    inline bool check_valid() const{
        return (row >= 0) && (col >= 0);
    }

    /*
        The getter and setter won't use the style:
        exception_t get/set(idx ... , value& somthing)
        because this will make the coding process for sparse matrix annoying
    */

    /*
        get value of row
    */
    inline index_t getrow() const{
        return row;
    }

    /*
        get value of col
    */
    inline index_t getcol() const{
        return col;
    }

    /*
        get value of data
    */
    inline value_t getdata() const{
        return data;
    }

    inline void setdata(value_t data){
        this->data = data;  
    }

};
