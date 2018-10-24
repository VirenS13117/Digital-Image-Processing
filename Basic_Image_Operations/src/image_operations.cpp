/*
 * add.cpp
 *      Author: virendra singh
 */

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>

using namespace cv;
int i,j;
int Rows, Cols;
int MaxRows, MaxCols;
void imageAdd(Mat image,Mat image2, Mat image3){
	for(i=0;i<Rows;i++){
		for(j=0;j<Cols*3;j++){
			if((image.at<uchar>(i,j) + image2.at<uchar>(i,j))>255){
				image3.at<uchar>(i,j)=255;
			}
			else if((image.at<uchar>(i,j) + image2.at<uchar>(i,j))<0){
				image3.at<uchar>(i,j)=0;
			}
			else{
				image3.at<uchar>(i,j) = image.at<uchar>(i,j) + image2.at<uchar>(i,j) ;
			}
		}
	}
	printf("Image Addition\n");
 }
 
void imageSub(Mat image,Mat image2, Mat image3){
	for(i=0;i<Rows;i++){
		for(j=0;j<Cols*3;j++){
			if((image.at<uchar>(i,j) - image2.at<uchar>(i,j))>255){
				image3.at<uchar>(i,j)=255;
			}
			else if((image.at<uchar>(i,j) - image2.at<uchar>(i,j))<0){
			    image3.at<uchar>(i,j)=0;
			}
			else{
			    image3.at<uchar>(i,j) = image.at<uchar>(i,j) - image2.at<uchar>(i,j) ;
			}
		}
	}
	printf("Absolute difference of two images\n");
 }

void complementImage(Mat image,Mat image2, Mat image3){
	for(i=0;i<Rows;i++){
			for(j=0;j<Cols*3;j++){
				 image3.at<uchar>(i,j) = 255 - image2.at<uchar>(i,j);
			}
		}
		printf("Complement of Image\n");
 }

void divideImage(Mat image,Mat image2, Mat image3){
	int constant;
	printf("enter a constant to divide the image\n");
	scanf("%d",&constant);
	for(i=0;i<Rows;i++){
	 	for(j=0;j<Cols*3;j++){
	 		if((image2.at<uchar>(i,j)/constant)>255){
	 			image3.at<uchar>(i,j)=255;
	 		}
	 		else if((image2.at<uchar>(i,j)/constant)<0){
	 		    image3.at<uchar>(i,j)=0;
	 		}
	 		else{
	 		    image3.at<uchar>(i,j) = image2.at<uchar>(i,j)/constant;
	 		}
	 	}
	}
	printf("Division of Image\n");
 }

void multiplyImage(Mat image,Mat image2, Mat image3){
	int constant;
	printf("enter a constant to multiply the image\n");
	scanf("%d",&constant);
	for(i=0;i<Rows;i++){
		for(j=0;j<Cols*3;j++){
			if((image2.at<uchar>(i,j)*constant)>255){
			 	image3.at<uchar>(i,j)=255;
			}
			else if((image2.at<uchar>(i,j)*constant)<0){
			    image3.at<uchar>(i,j)=0;
			}
			else{
			     image3.at<uchar>(i,j) = image2.at<uchar>(i,j)*constant;
			}
		}
	}
	printf("Multiplication of Image\n");
}

void linearCombination(Mat image,Mat image2, Mat image3){
	int constant1, constant2;
	printf("enter constants\n");
	scanf("%d%d",&constant1,&constant2);
	for(i=0;i<Rows;i++){
	    for(j=0;j<Cols*3;j++){
	 	 	//image2.at<uchar>(i,j) = constant1*image.at<uchar>(i,j) + constant2*image2.at<uchar>(i,j);
	 	 	if((constant1*image.at<uchar>(i,j) + constant2*image2.at<uchar>(i,j))>255){
	 	 		image3.at<uchar>(i,j)=255;
	 	 	}
	 	 	else if((constant1*image.at<uchar>(i,j) + constant2*image2.at<uchar>(i,j))<0){
	 	 	    image3.at<uchar>(i,j)=0;
	 	 	}
	 	 	else{
	 	 	    image3.at<uchar>(i,j) = constant1*image.at<uchar>(i,j) + constant2*image2.at<uchar>(i,j);
	 	 	}
	 	}
	}
	printf("Linear Combination of Image\n");
}
   
void absoluteDifference(Mat image, Mat image2, Mat image3){
    for(i=0;i<Rows;i++){
	  	for(j=0;j<Cols*3;j++){
	  		if((image.at<uchar>(i,j) - image2.at<uchar>(i,j))<0){
	  			image3.at<uchar>(i,j)= image2.at<uchar>(i,j)- image.at<uchar>(i,j);
	  		}
	  		else{
		  			image3.at<uchar>(i,j) = image.at<uchar>(i,j) - image2.at<uchar>(i,j);
	  			}
	  		}
	  	}
	  	printf("Absolute difference of two images\n");
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
 printf("1. Addition of two images\n");
 printf("2. Subtraction of one image from another\n");
 printf("3. Complement of image\n");
 printf("4. Divide image by constant\n");
 printf("5. Multiply image by constant\n");
 printf("6. Linear Combination of two images\n");
 printf("7. Absolute difference of two images\n");
 printf("Enter Your choice\n");
 scanf("%c",&ch);
 switch(ch){
 	 case '1':
 		 imageAdd(image, image2, image3);
 		 break;
 	 case '2':
 		 imageSub(image, image2, image3);
 		 break;
 	 case '3':
 		 complementImage(image, image2, image3);
 		 break;
 	 case '4':
 		 divideImage(image, image2, image3);
 		 break;
 	 case '5':
 		 multiplyImage(image, image2, image3);
 		 break;
 	 case '6':
 		 linearCombination(image, image2, image3);
 		 break;
 	 case '7':
 		 absoluteDifference(image, image2, image3);
 		 break;
 	 default:
 		 printf("Invalid Input\n");
 		 break;
     }

	 namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
	 imshow( "Display Image", image3 );
	 waitKey(0);

 	  	//imwrite("../../img.jpg",image2);

 return 0;
}
