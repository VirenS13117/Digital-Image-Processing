/*
 * Q1.cpp
 *
 *  Created on: 29-Aug-2015
 *      Author: virendra
 */


#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265

using namespace cv;
using namespace std;

int i,j,count=0;
double affineMatrixForRotation[3][3];
double affineMatrixForScaling[3][3];
double affineMatrixForTranslation[3][3];
double affineMatrix[3][3];
int ResultantMatrix[3];

/*
float DegreesToRadians(float degrees)
{
    float radians = degrees*M_PI/180;
    return radians;
}
*/
double standard_deviation(double data[], int n)
{
    double mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);
}

void enterAffineForRotation(){
	double angle;
	printf("Affine Matrix is of the form:\n");
	printf("[cos(X),-sin(X),Tx]\n");
	printf("[sin(X), cos(X),Ty]\n");
	printf("[0     , 0     ,1 ]\n");
	//int affineMatrix[3][3];
	printf("Enter the angle\n");
	scanf("%lf",&angle);
	affineMatrixForRotation[0][0] = cos(angle * PI / 180.0 );
	affineMatrixForRotation[0][1] = -1*sin(angle * PI / 180.0 );
	affineMatrixForRotation[1][0] = sin(angle * PI / 180.0 );
	affineMatrixForRotation[1][1] = cos(angle * PI / 180.0 );
	//printf("%f",affineMatrix[0][0]);
	//printf("Enter Tx\n");
	affineMatrixForRotation[0][2]=0;
	affineMatrixForRotation[1][2]=0;
	affineMatrixForRotation[2][0]=0;
	affineMatrixForRotation[2][1]=0;
	affineMatrixForRotation[2][2]=1;
}

void enterAffineForScaling(){
	printf("Affine Matrix is of the form:\n");
	printf("[a, b, Tx]\n");
	printf("[c, d, Ty]\n");
	printf("[0, 0, 1 ]\n");
	printf("Enter a\n");
	scanf("%lf",&affineMatrixForScaling[0][0]);
	printf("Enter b\n");
	scanf("%lf",&affineMatrixForScaling[0][1]);
	printf("Enter c\n");
	scanf("%lf",&affineMatrixForScaling[1][0]);
	printf("Enter d\n");
	scanf("%lf",&affineMatrixForScaling[1][1]);
	affineMatrixForScaling[0][2] = 0;
	affineMatrixForScaling[1][2] = 0;
	affineMatrixForScaling[2][0] = 0;
	affineMatrixForScaling[2][1] = 0;
	affineMatrixForScaling[2][2] = 1;
}

void enterAffineForTranslation(){
	printf("Affine Matrix is of the form:\n");
	printf("[a, b, Tx]\n");
	printf("[c, d, Ty]\n");
	printf("[0, 0, 1 ]\n");
	printf("Enter Tx\n");
	scanf("%lf",&affineMatrixForTranslation[0][2]);
	printf("Enter Ty\n");
	scanf("%lf",&affineMatrixForTranslation[1][2]);

	affineMatrixForTranslation[0][0] = 1;
	affineMatrixForTranslation[0][1] = 0;
	affineMatrixForTranslation[1][0] = 0;
	affineMatrixForTranslation[1][1] = 1;
	affineMatrixForTranslation[2][0] = 0;
	affineMatrixForTranslation[2][1] = 0;
	affineMatrixForTranslation[2][2] = 1;
}

void enterAffine(){
	double scale,angle;
	printf("Affine Matrix is of the form:\n");
	printf("Enter Scaling Factor\n");
	scanf("%lf",&scale);
	printf("Enter Angle\n");
	scanf("%lf",&angle);
	printf("Enter Translation in x-axis");
	scanf("%lf",&affineMatrix[0][2]);
	printf("Enter Translationin y-axis");
	scanf("%lf", &affineMatrix[1][2]);
	affineMatrix[0][0] = scale*cos(angle * PI / 180.0 );
	affineMatrix[0][1] = -1*scale*sin(angle * PI / 180.0 );
	affineMatrix[1][0] = scale*sin(angle * PI / 180.0 );
	affineMatrix[1][1] = scale*cos(angle * PI / 180.0 );
	affineMatrix[2][0] = 0;
	affineMatrix[2][1] = 0;
	affineMatrix[2][2] = 1;
}

void imageAffineForRotation(Mat image, Mat image3){
	int numberOfChannels = image.channels();
	Mat affine = Mat(3,3,CV_64F,affineMatrixForRotation);
	for(i=0;i<image.rows;i++){
		for(j=0;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){
				double pa[] = {(double)i,(double)j, 1};
				Mat pm = Mat(3, 1, affine.type(), pa);
				Mat npm = Mat(3, 1, affine.type());
				npm = affine*pm;
				double x = npm.at<double>(0,0);
				double y = npm.at<double>(1,0);

				if(x>0 && x<image.rows && y>0 && y<image.cols){
					image3.at<Vec3b>(y,x)[k] = image.at<Vec3b>(j, i)[k];
				}
			}
		}
	}

	imshow("rotated image", image3);
}

Mat imageSub(Mat image,Mat image2, Mat image3){
	int numberOfChannels = image.channels();
	 for(i=0;i<image.rows;i++){
		 for(j=0;j<image.cols;j++){
			 for(int k=0;k<numberOfChannels;k++){
				 if((image.at<Vec3b>(i,j)[k] - image2.at<Vec3b>(i,j)[k])>255){
					 image3.at<Vec3b>(i,j)[k]=255;
				 }
				 else if((image.at<Vec3b>(i,j)[k] - image2.at<Vec3b>(i,j)[k])<0){
					 image3.at<Vec3b>(i,j)[k]=0;
				 }
				 else{
					 image3.at<Vec3b>(i,j)[k] = image.at<Vec3b>(i,j)[k] - image2.at<Vec3b>(i,j)[k] ;
				 }
			 }
		 }
	 }
	 return image3;
 }


void imageAffineForScaling(Mat image, Mat image3){
	int numberOfChannels = image.channels();
	Mat affine = Mat(3,3,CV_64F,affineMatrixForScaling);
		for(i=0;i<image.rows;i++){
			for(j=0;j<image.cols;j++){
				for(int k=0;k<numberOfChannels;k++){


				//Point n = Point (i,j);
					double pa[] = {(double) (i), (double)( j), 1};
					Mat pm = Mat(3, 1, affine.type(), pa);

			//	printf("\n");
					Mat npm = Mat(3, 1, affine.type());

					npm = affine*pm;
					double x = npm.at<double>(0,0);
					double y = npm.at<double>(1,0);
					//printf("x y %f %f\n",x,y );
					if(x>0 && x<image.rows && y>0 && y<image.cols){
						image3.at<Vec3b>(x, y)[k] = image.at<Vec3b>(i,j)[k];
					}
				}
			}
		}

		imshow("image scaled",image3);
}

void translationOfImage(Mat image, Mat image3){
	printf("Enter AffineMatrix\n");
	enterAffineForTranslation();
	int numberOfChannels = image.channels();
	Mat affine = Mat(3,3,CV_64F,affineMatrixForTranslation);
			for(i=0;i<image.rows;i++){
				for(j=0;j<image.cols;j++){
					for(int k=0;k<numberOfChannels;k++){


					//Point n = Point (i,j);
						double pa[] = {(double) (i), (double)( j), 1};
						Mat pm = Mat(3, 1, affine.type(), pa);

				//	printf("\n");
						Mat npm = Mat(3, 1, affine.type());

						npm = affine*pm;
						double x = npm.at<double>(0,0);
						double y = npm.at<double>(1,0);
						//printf("x y %f %f\n",x,y );
						if(x>0 && x<image.rows && y>0 && y<image.cols){
							image3.at<Vec3b>(x, y)[k] = image.at<Vec3b>(i,j)[k];
						}
					}
				}
			}

			imshow("image translated",image3);
}

void RotationOfImage(Mat image, Mat image3){
	printf("Enter AffineMatrix\n");
	enterAffineForRotation();


	 printf("1. Performing Translation, Scaling and Rotation using affine matrix\n");
	 imageAffineForRotation(image, image3);
	 //imshow("Rotated image",image3);
}

Mat ScalingOfImage(Mat image, Mat image3){
	printf("Enter AffineMatrix\n");
    enterAffineForScaling();
    imageAffineForScaling(image, image3);
    return image3;
}

void affineTransform(Mat image, Mat image3){
	printf("Enter AffineMatrix\n");
	enterAffine();
	int numberOfChannels = image.channels();
	Mat affine = Mat(3,3,CV_64F,affineMatrix);
	for(i=0;i<image.rows;i++){
		for(j=0;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){
						//Point n = Point (i,j);
				double pa[] = {(double) (i), (double)( j), 1};
				Mat pm = Mat(3, 1, affine.type(), pa);
				//	printf("\n");
				Mat npm = Mat(3, 1, affine.type());
				npm = affine*pm;
				double x = npm.at<double>(0,0);
				double y = npm.at<double>(1,0);
				//printf("x y %f %f\n",x,y );
				if(x>0 && x<image.rows && y>0 && y<image.cols){
					image3.at<Vec3b>(y, x)[k] = image.at<Vec3b>(j,i)[k];
				}
			}
		}
	}

				imshow("image Operations all in one",image3);
		//the order in which translation, scaling and rotation is performed, makes the ouput different.
}

void imageHistogram(Mat image){
	int bins = 500;
	int numberOfChannels = image.channels();
	vector<Mat> histogram(numberOfChannels);
	vector<Mat> canvas(numberOfChannels);
	int maxValue[3]={0,0,0};
	for(i=0;i<histogram.size();i++){
		histogram[i] = Mat::zeros(1,bins,CV_32SC1);
	}
	printf("NumberOfChannels %d\n",numberOfChannels);
	for(i=0;i<image.rows;i++){
		for(j=0;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){
				uchar pixelF = (numberOfChannels == 1)?image.at<uchar>(i,j):image.at<Vec3b>(i,j)[k];
				histogram[k].at<int>(pixelF)+=1;
			}
		}
	}
	for(i=0;i<numberOfChannels;i++){
		for(j=0;j<bins-1;j++){
			if(histogram[i].at<int>(j) > maxValue[i]){
				maxValue[i] = histogram[i].at<int>(j);
			}
		}
	}
	const char* channelName[3] = { "blue", "green", "red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	for ( i = 0; i < numberOfChannels; i++)
	{
	    canvas[i] = Mat::ones(400, bins, CV_8UC3);
	    int rows;
	    for (j = 0, rows = canvas[i].rows; j < bins-1; j++)
	    {
	        line(
	            canvas[i],
	            Point(j, rows),
	            Point(j, rows - (histogram[i].at<int>(j) * rows/maxValue[i])),
	            numberOfChannels == 1 ? Scalar(200,200,200) : colors[i],
	            1, 8, 0
	        );
	    }

	    imshow(numberOfChannels == 1 ? "value" : channelName[i], canvas[i]);
	}
	waitKey(0);
	for(i=0;i<numberOfChannels;i++){
			for(j=1;j<bins-1;j++){
					histogram[i].at<int>(j)+=histogram[i].at<int>(j-1);

			}
		}
	for(i=0;i<numberOfChannels;i++){
			for(j=0;j<bins-1;j++){
				if(histogram[i].at<int>(j) > maxValue[i]){
					maxValue[i] = histogram[i].at<int>(j);
				}
			}
		}
	int mean = histogram[0].at<int>(bins-2)/bins;
	double histArray[bins-1];
	for(i=0;i<bins-1;i++){
		histArray[i] = histogram[0].at<int>(i);
	}
	if((mean)>127){
		printf("Light Image\n");
	}
	else{
		printf("Dark Image\n");
	}
	double std = standard_deviation( histArray, bins);
	if(mean/std > 0.5){
		printf("High Contrast\n");
	}
	else{
		printf("Low Contrast\n");
	}
	for ( i = 0; i < numberOfChannels; i++)
		{
		    canvas[i] = Mat::ones(600, bins, CV_8UC3);
		    int rows;
		    for (j = 0, rows = canvas[i].rows; j < bins-1; j++)
		    {
		        line(
		            canvas[i],
		            Point(j, rows),
		            Point(j, rows - (histogram[i].at<int>(j) * rows/maxValue[i])),
		            numberOfChannels == 1 ? Scalar(200,200,200) : colors[i],
		            1, 8, 0
		        );
		    }

		    imshow(numberOfChannels == 1 ? "CDF" : channelName[i], canvas[i]);
		}

}

void gammaTransform(Mat image){
	double c, gamma;
	int numberOfChannels = image.channels();
	printf("Gamma Transform : cr^(gamma)\n");
	printf("Enter the value of c and gamma\n");
	scanf("%lf%lf", &c, &gamma);
	for(i=0;i<image.rows;i++){
		for(j=0;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){

				double newValue = c*pow((double)image.at<Vec3b>(i,j)[k]/255,gamma);
				image.at<Vec3b>(i,j)[k] = newValue*255;

			}
			printf("\n");
		}
	}
	//namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
	imshow( "Gamma Transformed", image );

}

void imageNegative(Mat image){
	int numberOfChannels = image.channels();
	printf("Image Negative\n");
	for(i=0;i<image.rows;i++){
			for(j=0;j<image.cols;j++){
				for(int k=0;k<numberOfChannels;k++){
					image.at<Vec3b>(i,j)[k] = 255 - image.at<Vec3b>(i,j)[k];

				}
				printf("\n");
			}
		}
	imshow( "Image Negative", image );
}

int formulaContrastStretching(int x, int c, int a, int d, int b){
	float result;
		    if(0 <= x && x <= c){
		        result = a/c * x;
		    }else if(c < x && x <= d){
		        result = ((b - a)/(d - c)) * (x - c) + a;
		    }else if(d < x && x <= 255){
		        result = ((255 - b)/(255 - d)) * (x - d) + b;
		    }
		    return (int)result;
}

void contrastStretching(Mat image){
	int numberOfChannels = image.channels();
	int a,b,c,d;
	printf("Formula For contrast stretching\n");
	printf("s=(r-c)*(b-a) + a\n");
	printf("  -----------\n");
	printf("   (d-c)\n");
	printf("Enter a, b, c, d \n");
	scanf("%d%d%d%d",&a,&b,&c,&d);
	for(i=0;i<image.rows;i++){
		for(j=0;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){
				int s = formulaContrastStretching(image.at<Vec3b>(i,j)[k],c, a, d, b);
				image.at<Vec3b>(i,j)[k] = s;
			}
			printf("\n");
		}
	}
	imshow( "Contrast Stretching", image );
}

void averageFilter(Mat image){
	int numberOfChannels = image.channels();
	printf("Average Filter\n");
	printf("    [1,1,1]\n");
	printf("1/9*[1,1,1]\n");
	printf("    [1,1,1]\n");
	Mat image2 = Mat::zeros(image.rows, image.cols, image.type());
	for(i=1;i<image.rows;i++){
		for(j=1;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){
					int s = image.at<Vec3b>(i,j)[k]+image.at<Vec3b>(i-1,j-1)[k]+image.at<Vec3b>(i-1,j)[k]+image.at<Vec3b>(i-1,j+1)[k]+image.at<Vec3b>(i,j-1)[k]+image.at<Vec3b>(i,j+1)[k]+image.at<Vec3b>(i+1,j)[k]+image.at<Vec3b>(i+1,j-1)[k]+image.at<Vec3b>(i+1,j+1)[k];
					image2.at<Vec3b>(i,j)[k] = s/9;
			}
			printf("\n");
		}
	}
	imshow( "AverageFiltered", image2 );
}

void averageFilterFive(Mat image){
	int numberOfChannels = image.channels();
	printf("Average Filter\n");
	printf("    [1,1,1]\n");
	printf("1/25*[1,1,1]\n");
	printf("    [1,1,1]\n");
	Mat image2 = Mat::zeros(image.rows, image.cols, image.type());
	for(i=2;i<image.rows;i++){
		for(j=2;j<image.cols;j++){
			for(int k=0;k<numberOfChannels;k++){
					int s = image.at<Vec3b>(i,j)[k]+
							image.at<Vec3b>(i-1,j-1)[k]+
							image.at<Vec3b>(i-1,j)[k]+
							image.at<Vec3b>(i-1,j+1)[k]+
							image.at<Vec3b>(i,j-1)[k]+
							image.at<Vec3b>(i,j+1)[k]+
							image.at<Vec3b>(i+1,j)[k]+
							image.at<Vec3b>(i+1,j-1)[k]+
							image.at<Vec3b>(i+1,j+1)[k]+
							image.at<Vec3b>(i+2,j)[k]+
							image.at<Vec3b>(i+2,j+1)[k]+
							image.at<Vec3b>(i+2,j+2)[k]+
							image.at<Vec3b>(i-2,j)[k]+
							image.at<Vec3b>(i-2,j+1)[k]+
							image.at<Vec3b>(i-2,j+2)[k]+
							image.at<Vec3b>(i-1,j+2)[k]+
							image.at<Vec3b>(i+1,j+2)[k]+
							image.at<Vec3b>(i-2,j-2)[k]+
							image.at<Vec3b>(i-2,j-1)[k]+
							image.at<Vec3b>(i-1,j-2)[k]+
							image.at<Vec3b>(i,j-2)[k]+
							image.at<Vec3b>(i+1,j-2)[k]+
							image.at<Vec3b>(i+2,j-2)[k]+
							image.at<Vec3b>(i+2,j-1)[k]+
							image.at<Vec3b>(i,j+2)[k];
					image2.at<Vec3b>(i,j)[k] = s/25;
			}
			printf("\n");
		}
	}
	imshow( "AverageFiltered5x5", image2 );
}

void laplacian1(Mat image){
	int numberOfChannels = image.channels();
		printf("Average Filter\n");
		printf("    [0, 1, 0 ]\n");
		printf("	[1,-4, 1 ]\n");
		printf("    [0, 1, 0 ]\n");
		Mat image2 = Mat::zeros(image.rows, image.cols, image.type());
		Mat image4 = Mat::zeros(image.rows, image.cols, image.type());
		for(i=1;i<image.rows-1;i++){
			for(j=1;j<image.cols-1;j++){
				for(int k=0;k<numberOfChannels;k++){
						double s = (image.at<Vec3b>(i,j)[k])*(-4)+
								image.at<Vec3b>(i-1,j-1)[k]*0+
								image.at<Vec3b>(i-1,j)[k]*1+
								image.at<Vec3b>(i-1,j+1)[k]*0+
								image.at<Vec3b>(i,j-1)[k]*1+
								image.at<Vec3b>(i,j+1)[k]*1+
								image.at<Vec3b>(i+1,j)[k]*1+
								image.at<Vec3b>(i+1,j-1)[k]*0+
								image.at<Vec3b>(i+1,j+1)[k]*0;
						if(s<0){
							s=0;
						}
						else if(s>255){
							s=255;
						}
						image2.at<Vec3b>(i,j)[k] = abs(s);

						//image4.at<Vec3b>(i,j)[k] = s;
				}
				printf("\n");
			}
		}
		Mat image3 = Mat::zeros(image.rows, image.cols, image.type());
		image3 = imageSub(image, image2, image3);
		imshow( "Original Image", image );
		imshow( "Laplacian filtered", image2 );
		imshow(" Enhanced image", image3);

}

void laplacian2(Mat image){
	int numberOfChannels = image.channels();
		printf("Average Filter\n");
		printf("    [0, 1, 0 ]\n");
		printf("	[1,-4, 1 ]\n");
		printf("    [0, 1, 0 ]\n");
		Mat image2 = Mat::zeros(image.rows, image.cols, image.type());
		for(i=1;i<image.rows-1;i++){
			for(j=1;j<image.cols-1;j++){
				for(int k=0;k<numberOfChannels;k++){
						double s = (image.at<Vec3b>(i,j)[k])*(4)+
								image.at<Vec3b>(i-1,j-1)[k]*0+
								image.at<Vec3b>(i-1,j).val[k]*(-1)+
								image.at<Vec3b>(i-1,j+1)[k]*0+
								image.at<Vec3b>(i,j-1)[k]*(-1)+
								image.at<Vec3b>(i,j+1)[k]*(-1)+
								image.at<Vec3b>(i+1,j).val[k]*(-1)+
								image.at<Vec3b>(i+1,j-1)[k]*0+
								image.at<Vec3b>(i+1,j+1)[k]*0;
						if(s<0){
							s=0;
						}
						else if(s>255){
							s=255;
						}
						image2.at<Vec3b>(i,j)[k] = abs(s);
				}
				printf("\n");
			}
		}
		Mat image3 = Mat::zeros(image.rows, image.cols, image.type());
		imageSub(image, image2, image3);
		imshow( "Original Image", image );
		imshow( "Laplacian filtered", image2 );
		imshow(" Enhanced imageL2", image3);

}

int main(int argc, char** argv){
 char ch;
 Mat image, image2,gray;
 image = imread(argv[1],1);
 if( argc != 2 || !image.data )
         {
           printf( "No image data \n" );
           return -1;
         }
  cvtColor(image, gray, CV_BGR2GRAY);
  printf("%d\n",image.type());
  Mat image3 = Mat::zeros(image.rows, image.cols, image.type());
  printf("1. Rotation of image by affine Matrix\n");
  printf("2. Scaling of image by affine Matrix\n");
  printf("3. Translation of an image by affine Matrix\n");
  printf("4. Affine Transformation of image\n");
  printf("5. Gamma Transform of image\n");
  printf("6. Image Negative\n");
  printf("7. Histogram\n");
  printf("8. Contrast Stretching\n");
  printf("9. Average Filter\n");
  printf("A. Average Filter 5x5\n");
  printf("B. Laplacian Filter 1\n");
  printf("C. Laplacian Filter 2\n");
  printf("Enter Your Choice\n");
  scanf("%c",&ch);
  switch(ch){
 	 case '1':
 		 RotationOfImage(image, image3);
 		 break;
 	 case '2':
 		 ScalingOfImage(image, image3);
 		 break;
 	 case '3':
 		 translationOfImage(image,image3);
 		 break;
 	 case '4':
 		 affineTransform(image, image3);
 		 break;
 	 case '5':
 		 gammaTransform(image);
 		 break;
 	 case '6':
 		 imageNegative(image);
 		 break;
 	 case '7':
 		imshow("Original image",image);
 		imageHistogram(image);
 		break;
 	 case '8':
 		 contrastStretching(image);
 		 break;
 	 case '9':
 		 averageFilter(image);
 		 break;
 	 case 'A':
 		 averageFilterFive(image);
 		 break;
 	 case 'B':
 		 laplacian1(image);
 		 break;
 	 case 'C':
 	 	 laplacian2(image);
 	 	 break;
 	 default:
 	 		 printf("Invalid Input\n");
 	 		 break;
 	     }

 int newCols = image3.cols;
 int newRows = image3.rows;
 printf("Rows %d\n",newRows);
 printf("Cols %d\n",newCols);
 namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
 imshow( "Display Image", image3 );
 waitKey(0);
 return 0;
}
