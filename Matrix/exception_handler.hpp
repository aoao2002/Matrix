#pragma once
#include<iostream>
#include<cstdio>

class exception_handler{
public:
    /*
        puts red characters like:
        puts_err("114514");
    */
    static void puts_err(const char* str){
        std::string cppstr(str);
        cppstr = "\033[31m" + cppstr + "\033[0m";
        std::cout<<cppstr<<std::endl; 
    }

    /*
        puts red characters like:
        string s = "1919"; 
        puts_err(s);
    */
    static void puts_err(const std::string& str){
        std::string cppstr;
        cppstr = "\033[31m" + str + "\033[0m";
        std::cout<<cppstr<<std::endl; 
    }
};
