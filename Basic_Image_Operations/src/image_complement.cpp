/*
 * image_complement.cpp
 *      Author: virendra singh
 */

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
int i,j;
int Rows, Cols;
int MaxRows, MaxCols;

void complementImage(Mat image,Mat image2, Mat& image3){
	for(i=0;i<Rows;i++){
		for(j=0;j<Cols*3;j++){
			 image3.at<uchar>(i,j) = 255 - image2.at<uchar>(i,j);
		}
	}
	printf("Complement of Image\n");
}

int main(int argc, char** argv){
	char ch;
	Mat image, image2;
	image = imread(argv[1],1);
	image2 = imread(argv[2],1);
	Rows = MIN(image.rows,image2.rows);
	Cols = MIN(image.cols, image2.cols);
	MaxRows = MAX(image.rows,image2.rows);
	MaxCols = MAX(image.cols, image2.cols);
	Mat image3(MaxCols, MaxRows, CV_8UC3, Scalar(0, 0, 0));
	if( argc != 3 || !image.data )
	{
	    printf( "No image data \n" );
	    return -1;
	}
	complementImage(image, image2, image3);
	namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
	imshow( "Display Image", image3);
	waitKey(0);
	
	return 0;
}
