#include"matrix.hpp"
using namespace std;
int main(){
    
    double data[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    cv::Mat M(3, 3, CV_64F, data);
    for(int i = 0 ; i < M.rows ; i ++){
        for(int j = 0 ; j < M.cols ; j ++)
            cout<<M.at<double>(i,j)<<" ";
        cout<<endl;
    }
    cout<<endl;
    matrix<double> tmp = (matrix<double>)M;
    cout<<tmp<<endl;
    
    M = (cv::Mat)(tmp);
    for(int i = 0 ; i < M.rows ; i ++){
        for(int j = 0 ; j < M.cols ; j ++)
            cout<<M.at<float>(i,j)<<" ";
        cout<<endl;
    }
}