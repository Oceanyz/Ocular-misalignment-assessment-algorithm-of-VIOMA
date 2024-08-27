#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#define IMG_SIZE 200 //400

static cv::Mat delete_curves( cv::Mat *edge, double mean_dist){

	int start_x=0;
	int end_x=edge->cols;
	int start_y=0;
	int end_y=edge->rows;
	
	std::vector<std::vector<cv::Point>> all_lines;
	all_lines.clear();
	std::vector<cv::Point> all_means;
	all_means.clear();

	std::vector<cv::Point> curve; //用于临时存放检测到的curve
	int curve_idx=0;
	cv::Point mean_p;  //一段curve的平均位置

	bool delete_curve = false; //指示是否删除该段curve

	bool check[IMG_SIZE][IMG_SIZE]; //定义二维数组，初始化为零，这个二维数组的尺寸可以与图像的输入尺寸一致

	for(int i=0; i<IMG_SIZE; i++)
		for(int j=0; j<IMG_SIZE; j++)
			check[i][j]=0;
	
	//delete short and straight lines 
	for(int i=start_x; i<end_x; i++)
	{
		for(int j=start_y; j<end_y; j++)
		{
			if(edge->data[(edge->cols*(j))+(i)]>0 && !check[i][j]) //检查边缘轮廓点且未在check数组中置为1的点，若check中该位置点已置为1，则不用再看了
			{
				int neig=0; //初始化该点的非零邻居为0个
				for(int k1=-1;k1<2;k1++)
				{
					for(int k2=-1;k2<2;k2++)
					{
						if(i+k2>=start_x && i+k2<end_x && j+k1>=start_y && j+k1<end_y) //判断邻居点没有超出图像边界
						{
							if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
								neig++;
						}
					}
				}//计算该点9宫格中的非零邻居点
				if(neig>=2)
				{
					check[i][j]=1;

					curve.clear();
					curve_idx=0;

					curve.push_back(cv::Point(i,j)); //将有超过两个邻居的点压入curve
					mean_p.x=i;
					mean_p.y=j;
					curve_idx++;

					int akt_idx=0;

					while(akt_idx<curve_idx)
					{
						cv::Point akt_pos=curve[akt_idx];
						for(int k1=-1;k1<2;k1++)
						{
							for(int k2=-1;k2<2;k2++)
							{
								if(akt_pos.x+k1>=start_x && akt_pos.x+k1<end_x && akt_pos.y+k2>=start_y && akt_pos.y+k2<end_y) //判断邻居点没有超出图像边界
								{
									if(!check[akt_pos.x+k1][akt_pos.y+k2] )
									{
										if( edge->data[(edge->cols*(akt_pos.y+k2))+(akt_pos.x+k1)]>0)
										{
											check[akt_pos.x+k1][akt_pos.y+k2]=1;
											mean_p.x += akt_pos.x+k1;
											mean_p.y += akt_pos.y+k2;
											curve.push_back(cv::Point(akt_pos.x+k1,akt_pos.y+k2));
											curve_idx++;
										}
									}
								}
							
							}//内for
						}//外for
						akt_idx++;

					}//while

					if(curve_idx<=5 && curve.size()<=5)  //获得curve之后，执行两个判断，若curve很短，直接删掉
					{
						for(int r = 0;r<curve.size();r++)
						{
							int x = curve.at(r).x;
							int y = curve.at(r).y;

							edge->data[(edge->cols*(y))+(x)]=0; //将小于5个连续点（认为是孤立点），在边沿图像上置为0
							check[x][y]=0;
						}
					}//end if(curve_idx<=5 && curve.size()<=5)  
					
					if(curve_idx>5 && curve.size()>5) //进一步删除straight line
					{
						delete_curve = false;
						mean_p.x=floor(( double(mean_p.x)/double(curve_idx) )+0.5);
						mean_p.y=floor(( double(mean_p.y)/double(curve_idx) )+0.5);
						all_means.push_back(mean_p);//一个curve的平均位置
						all_lines.push_back(curve);//把curve压入all_lines
						
						int num = 0; //用来计算满足距离约束的曲线上点的个数
						float rate = 0; //rate是曲线上满足距离约束的点的比例
						for(int r = 0;r<curve.size();r++) //执行距离判断
						{
							if(  abs(mean_p.x-curve[r].x)<= mean_dist && abs(mean_p.y-curve[r].y) <= mean_dist) //判断curve上只要存在点到平均点的距离小于某个阈值，则认为是直线，然后执行删除操作
							{
								num++;
							}
						}
						rate = 1.0f*num/curve.size();
						if (rate > 0.2) //设置一个比率，避免由于眼睫毛等的干扰造成该曲线被删除 比率经验的设为0.005
						{
							delete_curve=true;
						}
						if(delete_curve) //执行删除straight line操作
						{
							for(int r = 0;r<curve.size();r++)
							{
								int x = curve.at(r).x;
								int y = curve.at(r).y;

								edge->data[(edge->cols*(y))+(x)]=0; //在边沿图像上置为0
								check[x][y]=0;
							}
						}
						
					}//end if(curve_idx>5 && curve.size()>5)
					
				}//if(neig>=2)
			}
		}//内for循环结束
	}//外for循环结束
	/*
	int size_lines = all_lines.size();
	if(size_lines<15)
	{
		for(int s = 0; s<all_lines.size();s++)
		{
			delete_curve = false;
			mean_p = all_means.at(s);
			curve = all_lines.at(s);
			int num = 0; //用来计算满足距离约束的曲线上点的个数
			float rate = 0; //rate是曲线上满足距离约束的点的比例
			for(int r = 0;r<curve.size();r++) //执行距离判断
			{
				if(  abs(mean_p.x-curve[r].x)<= mean_dist && abs(mean_p.y-curve[r].y) <= mean_dist) //判断curve上只要存在点到平均点的距离小于某个阈值，则认为是直线，然后执行删除操作,这样容易导致误删，比如眼睫毛深入到瞳孔里
				{
					num++;
				}
			}
			rate = 1.0f*num/curve.size();
			if (rate > 0.2) //设置一个比率，避免由于眼睫毛等的干扰造成该曲线被删除 比率经验的设为0.005,0.1,0.2
			{
				delete_curve=true;
			}
			if(delete_curve) //执行删除straight line操作
			{
				for(int r = 0;r<curve.size();r++)
				{
					int x = curve.at(r).x;
					int y = curve.at(r).y;

					edge->data[(edge->cols*(y))+(x)]=0; //在边沿图像上置为0
					check[x][y]=0;
				}
			}
		}
	}//end if(size_lines<100)
	*/
	//边沿细化处理
	for(int j=start_y; j<end_y; j++)
	{
		for(int i=start_x; i<end_x; i++)
		{
			int box[9];

			box[4]=(int)edge->data[(edge->cols*(j))+(i)];

			if(box[4])
			{
				if(j-1>=start_y && j+1<end_y && i-1>=start_x && i+1<end_x)
				{
					box[1]=(int)edge->data[(edge->cols*(j-1))+(i)];
					box[3]=(int)edge->data[(edge->cols*(j))+(i-1)];
					box[5]=(int)edge->data[(edge->cols*(j))+(i+1)];
					box[7]=(int)edge->data[(edge->cols*(j+1))+(i)];

					if((box[5] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
					if((box[5] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
					if((box[3] && box[7])) edge->data[(edge->cols*(j))+(i)]=0;
					if((box[3] && box[1])) edge->data[(edge->cols*(j))+(i)]=0;
					//细化边沿
				}
			}
		}
	}
	//too many neigbours具有超过3个邻居的边沿点被删掉，因为它倾向于连接多个边沿线
	for(int j=start_y; j<end_y; j++)
	{
		for(int i=start_x; i<end_x; i++)
		{
			int neig=0;
			for(int k1=-1;k1<2;k1++)
			{
				for(int k2=-1;k2<2;k2++)
				{
					if(i+k2>=start_x && i+k2<end_x && j+k1>=start_y && j+k1<end_y) //判断邻居点没有超出图像边界
					{
						if(edge->data[(edge->cols*(j+k1))+(i+k2)]>0)
							neig++;
					}
				}
			}

			if(neig>3)
				edge->data[(edge->cols*(j))+(i)]=0;

		}//内for
	}//外for

	return *edge;
}
