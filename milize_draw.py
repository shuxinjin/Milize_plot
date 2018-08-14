# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:26:34 2018

@author: sxjin
"""

# import models

import matplotlib.pyplot as plt
from io import BytesIO

import base64

import numpy as np

# Method2 implemented
import imageio,os
from PIL import Image




def ttest():
    print("yes ,you are in milizeai AI kernal... ")
    #create_gif(CUR_PATH,CUR_PATH,html_gif_name)
    return "test123"
   
   
#################################


ainame="getAInameFromoutside"
passvalue=9999 #should get from outside,keep for future

#this name depend on caller, caller should know path and gif_name,all depend on caller,will not return .
#*******************html_gif_name="milizeAI.gif"
#******************CUR_PATH = r'c:/JINSHUXIN/Milize_AI/Milize_AI/templates/images/'

fileorder=1000000001


#clean the temp pic folder. If dont want others pics interfere with you , 
#you should clear the folder.
#DANGEROUS!!!!,make sure you want this action

def del_file(path):
    
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)






       

#******pgn buffer convert to base64 binary**#
#*******************problem ,need debug********************#
def png_buffer_to_base64html(fig,ainame, passvalue, *args):
    #begin
    #consider check security?

    pngbinary="" 
    #method 1, create html binary stream files.
    buffer = BytesIO()  #buffer 
    
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
     
     
     
    plot_data = buffer.getvalue()
    #imb = base64.b64encode(buffer.getvalue())  
    pngbinary = base64.encodebytes(plot_data).decode()
    buffer.close
    return pngbinary
    
    
    

#*********convert png files to base64 binary of html ********#  
    
def png_files_to_base64html(openpath, pngnam):
    #begin
    #consider check security?

   

    pngbinary="" 

    
    #if a gif name exist
    if pngnam:
        with open(openpath + pngnam,"rb") as fpng:
        
            pngbinary= base64.encodebytes(fpng.read()).decode()
        #create base64 binary for gif picture
        return pngbinary
   
        
    #gif name not exist
    else:
        return ""
    

    

#******pgn buffer convert to base64 binary**#
#consider args[0] as gif name
def gif_to_base64html(openpath, gifnam):
    gifbinary =""
    
    #if a gif name exist
    if gifnam:
        with open(openpath + gifnam,"rb") as fgif:
            # 
            gifbinary= base64.encodebytes(fgif.read()).decode()
        #create base64 binary for gif picture
        return gifbinary
   
        
    #gif name not exist
    else:
        return ""
#*******************************************# 



    
    
    

#************Abstract save png**********#
#***************************************#
## *args type (), **kw type {}
#figure,fig
#plot ,plt
#passvalue ,y_test, test Dataset
#args0 and 1 is the figsize para,
#args2 is Epoch , str(ee)
#args3 is Batch ,str(ii))
def save_to_png(fig,plt,datatest,fileorder,save_path,*args,**kw):
    #begin
    #use this buffer to store series legends,if more than 1.
    #for common pic
    preimg=""
   
  
    #optimize the chart
    #fig.set_dpi(100)
    #args to int
    list1=[]
    
    list1 = [int(x) for x in args]
 
    
    if (list1[0] and list1[1]):
        fig = plt.figure(figsize=(list1[0],list1[1]))
    else:
        #if this para not be got
        fig = plt.figure(figsize=(10,7.6))
        
    #clear the history mark???
    #plt.xticks([])  #去掉横坐标值
    #plt.yticks([])  #去掉纵坐标值
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    #plt.xticks(0, [1970,1980,1990,2000,2010,2020,2030])
    #ax1.set_yticks([0,500,1000,1500,2000,2500,3000,3500,4000])
    #ax1.set_xticks([1970,1980,1990,2000,2010,2020,2030])
    #ax1.xaxis_date()
    #plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    #plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    
    
    ax1 = fig.add_subplot(111)
    
    #ax1.xaxis.grid(True, which='minor') #x坐标轴的网格使用主刻度
    #ax1.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    
    #test dataset
    ax1.plot(datatest)
    ax1.plot(pred.transpose())
    
    #title of chart,ee jis epoch number, ii is batch number
    
    ee=list1[2]
   
    ii=list1[3]
   
    #list1 = map(lambda xx:type(xx)==int, args)
    #print("list you:"+ str(ee)+str(ii))
    plt.title('Epoch ' + str(ee) + ', Batch ' + str(ii))
    #print("list you:"+ str(ii)  +"title :"+'Epoch ' + str(ee) + ', Batch ' + str(ii))
    #pause for view
    plt.pause(0.01)
    
    

    
    preimg='epoch_' + str(fileorder)+'.png'
    
    fig.savefig(save_path+preimg, format='png')  
    #print("save_path+preimg:"+ save_path+preimg)
    
    #release
    save_path=""
    
    #args,kw=None
    
    #end of save to png, png folder still caller give the CUR_PATH
    
#***************************************#

    
    #release plt
    plt.close()
    
    
    
    
#******   create gif format pic  *******#
#***************************************#    
def create_gif(png_path,gif_path,gif_name):
    # *****  Method 2 
    images = []   
    ani_ims = []

    #print("images:"+str(images))

    #read all png files in the special folder
    filenames=sorted((fn for fn in os.listdir(png_path) if fn.endswith('.png')))
    #must sort the name,otherwise ,gif not in order 
    #print("filenames:"+str(filenames))
    for filename in filenames:
     
        #print("filename:"+str(filename))
        #method 2
        images.append(imageio.imread(png_path+filename))
        
        
        # be used by method 3 , TEMPLY keep here
        img1 = np.array(Image.open(png_path+filename))
        im1 = plt.imshow(img1, animated=True)
        ani_ims.append([im1])
        
    #create gif    
    #imageio type to create gif  ,method 2.1  
    imageio.mimsave(gif_path+gif_name, images,duration=1)

    return True
    #end , return a gif folder+name    
#***************************************#



#***************read all pngs together in this time
#***************convert to binary together 

def all_pngs_to_binary64(CUR_PATH1):
    
    iris_im=""
    
    iris_im = iris_im+"""<tr>"""
    ims=""
    tcount=0
    
    #read all png files in the special folder
    
    filenames1=sorted((fn1 for fn1 in os.listdir(CUR_PATH1) if fn1.endswith('.png')))
    #must sort the name,otherwise ,gif not in order 
    for filename1 in filenames1:  
        ims=png_files_to_base64html(CUR_PATH1, filename1)
    
        #ims= png_buffer_to_base64html(fig,ainame, passvalue)
        ims = "data:image/png;base64,"+ims
        #iris_im = iris_im+"""<td><h3> AI Predict Step: """ +'Epoch ' + str(e) + ', Batch ' + str(i)+ """</h3> <img src="%s">""" % ims + """<br></td>"""
        iris_im = iris_im+"""<td><h3> AI Predict Step>>> """ +str(tcount)+ """</h3> <img src="%s">""" % ims + """<br></td>"""
        tcount=tcount+1
        if ((tcount% 3)==0):
            iris_im = iris_im +"""</tr><tr>"""
        else:
            pass
        ims=""
    iris_im = iris_im +"""</tr>"""
    
    return iris_im
#**************************************************


















