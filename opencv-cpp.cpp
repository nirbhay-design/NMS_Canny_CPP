// opencv-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void print_mat(Mat mt) {
    int height = mt.size().height;
    int width = mt.size().width;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << mt.row(i).col(j) << " ";
        }
        cout << endl;
    }
}

void divide_matrix(Mat& mt, float divisior) {
    for (int i = 0; i < mt.rows; i++) {
        for (int j = 0; j < mt.cols; j++) {
            mt.row(i).col(j) = mt.row(i).col(j) / divisior;
        }
    }
}


Mat filter2d(Mat source, Mat kernel) {
    Size source_size = source.size();
    Size kernel_size = kernel.size();

    Size dest_size = Size(source_size.width - kernel_size.width + 1, source_size.height - kernel_size.height + 1);
    Mat dest_matrix(dest_size, CV_32FC1);

    for (int i = 0; i < dest_matrix.rows; i++) {
        for (int j = 0; j < dest_matrix.cols; j++) {
            int kh = kernel_size.height;
            int kw = kernel_size.width;

            float conv_val = 0;
            for (int l = i; l < i + kh; l++) {
                for (int m = j; m < j + kw; m++) {
                    conv_val += (float) source.at<uchar>(l,m) * kernel.at<float>((l - i), (m - j));
                }
            }
            dest_matrix.row(i).col(j) = conv_val;
        }
    }

    return dest_matrix;
    

}

pair<Mat, Mat> grad_magnitude_angle(Mat gx, Mat gy) {
    Size gx_size = gx.size();
    Size gy_size = gy.size();
    Mat g_mag(gx_size, CV_32FC1);
    Mat g_angle(gx_size, CV_32FC1);

    for (int i = 0; i < gx_size.height; i++) {
        for (int j = 0; j < gx_size.width; j++) {
            float gx_ij =  gx.at<float>(i,j);
            float gy_ij = gy.at<float>(i, j);
            g_mag.row(i).col(j) = sqrt(gx_ij * gx_ij + gy_ij * gy_ij);
            if (gy_ij == 0.0 && gx_ij == 0.0) {
                g_angle.row(i).col(j) = 0.0;
            }
            else {
                g_angle.row(i).col(j) = atan(gy_ij / gx_ij) * 180.0 / 3.14;
            }
        }
    }

    return make_pair(g_mag, g_angle);
}

void convert_uint8(Mat& mt) {
    int max_val = INT_MIN;
    for (int i = 0; i < mt.rows; i++) {
        for (int j = 0; j < mt.cols; j++) {
            uint8_t val = (uint8_t) mt.at<float>(i,j);
            mt.row(i).col(j) = val;

            if (val > max_val) {
                max_val = val;
            }
        }
    }

    for (int i = 0; i < mt.rows; i++) {
        for (int j = 0; j < mt.cols; j++) {
            uint8_t val = (255 / max_val) * (uint8_t)mt.at<float>(i, j);

            mt.row(i).col(j) = val;
        }
    }
}

bool valid_coordinates(int i, int j, int n, int m) {
    if (i < n && j < m && i >= 0 && j >= 0) {
        return true;
    }

    return false;
}

void NonMaxSupression(Mat& mt, Mat ang_mat) {
    cout << "inside NMS" << endl;
    for (int i = 0; i < mt.rows; i++) {
        for (int j = 0; j < mt.cols; j++) {
            pair<int,int> c1, c2;
            c1 = make_pair(i, j);
            c2 = make_pair(i, j);
            float angle = ang_mat.at<float>(i, j);
            float cur_val = mt.at<float>(i, j);

            if (angle <= 22.5 && angle > -22.5) {
                c1 = make_pair(i, j + 1);
                c2 = make_pair(i, j - 1);
            }
            else if (angle <= 67.5 && angle > 22.5) {
                c1 = make_pair(i - 1, j + 1);
                c2 = make_pair(i + 1, j - 1);
            }
            else if ((angle <= 91 && angle > 67.5) || (angle <= -67.5 && angle > -91)) {
                c1 = make_pair(i + 1, j);
                c2 = make_pair(i - 1, j);
            }
            else {
                c1 = make_pair(i - 1, j - 1);
                c2 = make_pair(i + 1, j + 1);
            }


            bool c1_valid = valid_coordinates(c1.first, c1.second, mt.rows, mt.cols);
            bool c2_valid = valid_coordinates(c2.first, c2.second, mt.rows, mt.cols);


            if (c1_valid && c2_valid) {
                float val_c1 = mt.at<float>(c1.first, c1.second);
                float val_c2 = mt.at<float>(c2.first, c2.second);

                if (cur_val < val_c1 || cur_val < val_c2) {
                    mt.row(i).col(j) = 0.0;
                }
            } else if (c1_valid && !c2_valid) {
                float val_c1 = mt.at<float>(c1.first, c1.second);

                if (cur_val < val_c1) {
                    mt.row(i).col(j) = 0.0;
                }
            } else if (!c1_valid && c2_valid) {
                float val_c2 = mt.at<float>(c2.first, c2.second);

                if (cur_val < val_c2) {
                    mt.row(i).col(j) = 0.0;
                }
            }
        }
    }
}

void thresholding(Mat& mt, int low, int high) {
    for (int i = 0; i < mt.rows; i++) {
        for (int j = 0; j < mt.cols; j++) {
            uint8_t cur_val = (uint8_t) mt.at<float>(i, j);
            if (cur_val < low) {
                mt.row(i).col(j) = 0;
            } 

            if (cur_val > high) {
                mt.row(i).col(j) = 255;
            }
        }
    }
}

int main()
{
    string path = "sample.jpg";

    Mat img = imread(path);

    Mat blurImg, cannyImg, grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    GaussianBlur(grayImg, blurImg, Size(3, 3), 3, 0);
    Canny(blurImg, cannyImg, 50, 150);

    Size sz = img.size();

    int width = sz.width;
    int height = sz.height;

    cout << width << " " << height << endl;

    float kx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    float ky[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    Mat k_x(3, 3, CV_32FC1, kx);
    Mat k_y(3, 3, CV_32FC1, ky);

    divide_matrix(k_x, 9);
    divide_matrix(k_y, 9);

    Mat gx, gy;
    Mat g_mag, g_ang;

    gx = filter2d(blurImg, k_x);
    gy = filter2d(blurImg, k_y);

    pair<Mat, Mat> G = grad_magnitude_angle(gx, gy);
    
    NonMaxSupression(G.first, G.second);

    convert_uint8(G.first);
    imshow("G_NMS", G.first);
    thresholding(G.first, 50, 150);
    imshow("G_thresh", G.first);
    imshow("cannyimg", cannyImg);

    convert_uint8(gx);
    convert_uint8(gy);

    imshow("Gx", gx);
    imshow("Gy", gy);
    imshow("Original Img", grayImg);

    waitKey(0);

    return 0;

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
