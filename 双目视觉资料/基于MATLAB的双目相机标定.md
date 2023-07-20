    基于双目视觉的测距、三维重建等过程中的第一步就是要进行标定，标定的精度对于整个双目系统性能的影响举足轻重。双目相机的标定过程在网上有很多资料，但是基本都没有matlab官方网址讲的好。所以请参考MATLAB官方文档：[Using the Stereo Camera Calibrator App- MATLAB & Simulink- MathWorks 中国](https://ww2.mathworks.cn/help/vision/ug/stereo-camera-calibrator-app.html)，这里面讲得已经相当详细了！

下面整理一下要点：

# 一、拍摄棋盘格

    为获得最佳效果，请使用至少10到20张包含校准图案的图像。 校准器至少需要三个图像。 尽量使用未压缩或压缩损失很小的图像格式（如png或bmp）。 为了更高的校准精度你需要：

-   获取一部分你所关注的距离处的棋盘格图片，比如你要测量2米远的物体，那么请将棋盘放在距离相机2米左右的地方拍摄一部分图片
    
-   棋盘表面和相机成像平面的夹角必须小于45度
    ![](https://img-blog.csdnimg.cn/20190828110640329.png)
    
-   不要修改图像，比如对其进行剪切
    
-   不要使用自动聚焦模式或改变图像的放大倍率
    
-   以相对于相机的不同方向拍摄棋盘图像
    
-   尽量采集各种不同的棋盘图像。镜头的畸变从图像中心径向增加，并且有时在图像各帧上表现不均匀， 为了获取图像的畸变信息，棋盘应当处在图像的各种不同边缘处
    ![](https://img-blog.csdnimg.cn/20190828111729381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1bGluZ3dlbg==,size_16,color_FFFFFF,t_70)
    
-   确保棋盘图案在左右两幅图像中都能被完整的显示
    ![](https://img-blog.csdnimg.cn/2019082811221434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1bGluZ3dlbg==,size_16,color_FFFFFF,t_70)
    
-   在每一对图像中尽量保持棋盘静止，也就是在同一时间拍摄。 若棋盘在两幅图像中发生了相对运动，会对标定精度产生负面影响
    
-   若想对远距离的重建获得更高的精度，需要将两个相机的距离调整的更大。
    

# 二、提升标定精度

## 1.添加或删除图像

添加图像:

-   少于10张图像
-   棋盘没有覆盖足够的图像帧
-   棋盘与相机的相对方向变化不够多

删除图像:

-   删除具有较大重投影误差的图像
-   图像太模糊的删除
-   棋盘平面与相机平面夹角超过45度的删除
    ![](https://img-blog.csdnimg.cn/20190828113138671.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1bGluZ3dlbg==,size_16,color_FFFFFF,t_70)
    
## 2.将3 coeefficients、Skew、Tangential Distortion全部勾选
![](https://img-blog.csdnimg.cn/20190828113530744.png)

# 三、标定结果应用到OpenCV

    通过matlab标定后得到的旋转矩阵R和内参矩阵K，都需要转置以后才可以给OpenCV用，另外OpenCV畸变向量中前5个畸变系数的依次是：\[k1, k2, p1, p2, k3\]