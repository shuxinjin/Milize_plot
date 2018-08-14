# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:26:34 2018

@author: sxjin
"""

# import models
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from lxml import etree
import base64

#import matplotlib.dates as mdate




# predict need
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#method 1
import webbrowser


# Method2 implemented
import imageio,os
from PIL import Image

#method 3
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter



def ttest():
    print("yes ,you are in milizeai AI kernal... ")
    #create_gif(CUR_PATH,CUR_PATH,html_gif_name)
    return "test123"
   

################## predict start ###############################

# Import dataset from csv file
data = pd.read_csv('C:/JINSHUXIN/Milize_AI/Milize_AI/sp500/data_stocks.csv')


# Drop date variable
# process the date colum. if 1 means column, if 0 means row .
#there are some special functions in pandas,if you need process such data format
#,you need learn the spection functions with it.
data = data.drop(['DATE'], 1)

# Dimensions of dataset
#matrix have one attribute ,column and row .Use shape() method to show the number.
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
# 80% data as the test dataset. 
# floor() returns the largest integer not greater than the input parameter
train_end = int(np.floor(0.8*n))

#test dataset
test_start = train_end + 1
test_end = n


#slice operation,
data_train = data[np.arange(train_start, train_end), :]

#target to return a ndarray type ,then slice them 
data_test = data[np.arange(test_start, test_end), :]



# Scale data  
# Most NN data need data normalization , because the data range of tanh and sigm characteristic
# transfer dataset between range of min and max
scaler = MinMaxScaler(feature_range=(-1, 1))

scaler.fit(data_train)

#calculate as scaler function defination.
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
#1 is the index of column, 1 to last column 
X_train = data_train[:, 1:]
#0 column data, this column is the Y target data
y_train = data_train[:, 0]

#test
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Number of stocks in training data
#shape 0 is matrix row number ,shape 1 is the column number .
n_stocks = X_train.shape[1]

# Neurons
#be used in below formula
n_neurons_1 = 1024

n_neurons_2 = 512

n_neurons_3 = 256

n_neurons_4 = 128


# Session
# apply interactive session in current project.
net = tf.InteractiveSession()


# Placeholder. place holder is the theroy .deployed the place holder

X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])

Y = tf.placeholder(dtype=tf.float32, shape=[None])


# Initializers

sigma = 1

#there are several kind of initializer type, this is one of them .
#a rd article ,https://blog.csdn.net/m0_37167788/article/details/79073070
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)

bias_initializer = tf.zeros_initializer()



# Hidden weights

W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))

bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))

bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))

bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))

bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))



# Output weights
# calculate the output.

W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))

bias_out = tf.Variable(bias_initializer([1]))



# Hidden layer .

#1. linear transformation for neural unit.
#2. tf.matmul(..., ...),Multiply the matrix, the number of left matrix 
#   columns is equal to the number of right matrix rows.
#3. tf.add(..., ...) —— addition
#4. np.random.randn(...)——  random init

hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))



# Output layer (transpose!)
# axis transpose. one article help you understand this ,
# https://blog.csdn.net/lothakim/article/details/79494782
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))



# Cost function

#The loss function of the network is mainly used to generate a deviation value 
#between the network prediction and the actually observed training target.
#For regression problems, the mean square error (MSE) function is most commonly used. 
#The MSE calculates the average squared error between the predicted and target values.

mse = tf.reduce_mean(tf.squared_difference(out, Y))




'''
The optimizer handles the necessary calculations for adapting to network weights 
and bias variables during training. These calculations call the gradient 
calculation results, indicating the direction in which the weights and 
deviations need to be changed during training, thereby minimizing the cost 
function of the network. The development of stable and fast optimizers has 
always been an important research in the field of neural networks and deep learning.



The above is the use of the Adam optimizer, which is the default optimizer 
in deep learning today. Adam stands for Adaptive Moment Estimation and can
 be used as a combination of the two optimizers AdaGrad and RMSProp.
'''
# Optimizer

opt = tf.train.AdamOptimizer().minimize(mse)


'''
The initializer is used to initialize the variables of the network before 
training. Because neural networks are trained using numerical optimization 
techniques, the starting point for optimization problems is the focus of 
finding a good solution. There are different initializers in TensorFlow,
 each with a different initialization method. In this article, I am using tf.
 variance_scaling_initializer(), which is a default initialization strategy.
'''
# Init

net.run(tf.global_variables_initializer())



# Setup plot

plt.ion()

fig = plt.figure()

ax1 = fig.add_subplot(111)

line1, = ax1.plot(y_test)

line2, = ax1.plot(y_test * 0.5)

plt.show()



# Fit neural net

batch_size = 256

mse_train = []

mse_test = []



################ predict config end   ################################


# update timely
#####iris_des = """<h1>Milize AI Predict US Stock </h1>"""+"""<meta http-equiv＝"refresh" content="5"> """
iris_des = """<h1>Milize AI Predict Partition </h1>"""+"""<script language="JavaScript"> setTimeout(function(){location.reload()},70000); </script>"""
 


# Run cycle of dataset
epochs = 2


#use this buffer to store series legends,if more than 1.
#for common pic
iris_im=""
#use for deliver parameters
imgnums=[]

#for gif format pic
iris_imgif=""
imsgif=""
ainame="getAInameFromoOutside"
passvalue=9999 #should get from outside,keep for future

#this name depend on caller, caller should know path and gif_name,all depend on caller,will not return .
html_gif_name="milizeAI.gif"
CUR_PATH = r'c:/JINSHUXIN/Milize_AI/Milize_AI/templates/images/'
# for png file temply save 
pngtmpfolder=CUR_PATH

#  Method 3 implement this ,with ffmpeg#
# Fixing random state for reproducibility


ani_ims = []



#clean the temp pic folder. If dont want others pics interfere with you , 
#you should clear the folder.




def del_file(path):
    
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)




#draw a figure of matplotlib, MUST be config to public parameter
fig = plt.figure()

fileorder=1000000001





          

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
    
    
    

#*******************************************#  
    
def png_files_to_base64html(openpath, pngnam):
    #begin
    #consider check security?

   

    pngbinary="" 

    
    #if a gif name exist
    if pngnam:
        with open(openpath + pngnam,"rb") as fpng:
            # b64encode是编码，b64decode是解码
            pngbinary= base64.encodebytes(fpng.read()).decode()
        #create base64 binary for gif picture
        return pngbinary
   
        
    #gif name not exist
    else:
        return ""
    

    

#******pgn buffer convert to base64 binary**#
#***** 2.2 *********************************#
#consider args[0] as gif name
def gif_to_base64html(openpath, gifnam):
    gifbinary =""
    
    #if a gif name exist
    if gifnam:
        with open(openpath + gifnam,"rb") as fgif:
            # b64encode是编码，b64decode是解码
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
    
    #release plt
    plt.close()

#***************************************#

    
    
    
    
    
    
    
    
    
#******   create gif format pic  *******#
#***************************************#    
def create_gif(png_path,gif_path,gif_name):
    # *****  Method 2 
    images = []    
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
        img1 = np.array(Image.open(CUR_PATH+filename))
        im1 = plt.imshow(img1, animated=True)
        ani_ims.append([im1])
        
    #create gif    
    #imageio type to create gif  ,method 2.1  
    imageio.mimsave(gif_path+gif_name, images,duration=1)

    return True
    #end , return a gif folder+name    
#***************************************#










#  dangerous!!!!!make sure,before the code cycle ,
#  Clear all png files of the temp folder
del_file(CUR_PATH)

      




#######

for e in range(epochs):


    
    # Shuffle training data

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    X_train = X_train[shuffle_indices]

    y_train = y_train[shuffle_indices]


    
    # Minibatch training
    
    for i in range(0, len(y_train) // batch_size):
        
        
        
        start = i * batch_size
        #slice
        batch_x = X_train[start:start + batch_size]

        batch_y = y_train[start:start + batch_size]

        # Run optimizer with batch
        # run the session 
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        #draw a table html format
     
        
         # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            
            
                

           
            #0 and 1 is the figsize para,
            #2 is Epoch , str(ee)
            #3 is Batch ,str(ii))
            imgnums=[10,7,e,i]
            #optimize the chart
            #fig.set_dpi(100)
            #设置输出的图片大小
            #print(str(imgnums))
            #def save_to_png(fig,plt,datatest,fileoder,save_path,*args,**kw,):
            fileorder=fileorder+1 #increase order outside of the func
            #***************important：
            #               if you want clear this image folder ,you should do it in your codes
            #               and clear all png files before your code cycle 
            save_to_png(fig,plt,y_test,fileorder,CUR_PATH,*imgnums)
            

                
            #one picture figure, the area of it be named axes:            
            ####fig,ax1 = plt.subplots(1,4,sharey = True)
            
            
            #****METHOD 3 ********************************#
            
            #fig, ax = plt.subplots()
            #method 3
            #ani = animation.FuncAnimation(fig, './templates/images/'+filename, blit=False, frames=200, interval=20, repeat=False)
            
    iris_im = iris_im +"""</tr>"""
 
 




#call this 
create_gif(CUR_PATH,CUR_PATH,html_gif_name)




iris_im=""

iris_im = iris_im+"""<tr>"""

#read all pngs together in this time
#convert to binary together 
ims=""
tcount=0
images1 = []    

#read all png files in the special folder
filenames1=sorted((fn1 for fn1 in os.listdir(CUR_PATH) if fn1.endswith('.png')))
#must sort the name,otherwise ,gif not in order 
for filename1 in filenames1:  
    ims=png_files_to_base64html(CUR_PATH, filename1)

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







# method 1 ,embeded one timely fresh script
root = "<title>Milize AI Predict Stock</title>"
root = root + iris_des + "<table>"+ iris_im+"</table>"  

# lxml lib etree convert the info to html soure code ,and write it to file
html = etree.HTML(root)
tree = etree.ElementTree(html)
tree.write('C:/JINSHUXIN/Milize_AI/Milize_AI/templates/predict2.html')

webbrowser.open('C:/JINSHUXIN/Milize_AI/Milize_AI/templates/predict2.html',new = 0.3)









# Method 2.1 , create one single gif html
root = "<title>Milize AI Predict Stock</title>"
#???????????????change the name path,later
ims111=""
ims111="images/" + html_gif_name
root = root + """ <br><img src="%s">""" % ims111 + """<br>"""
# lxml lib etree convert the info to html soure code ,and write it to file
html = etree.HTML(root)
tree = etree.ElementTree(html)
tree.write('C:/JINSHUXIN/Milize_AI/Milize_AI/templates/predict1.html')
# open html file with explorer
webbrowser.open('C:/JINSHUXIN/Milize_AI/Milize_AI/templates/predict1.html',new = 0.3)








# Method 2.2 ,call it
imsgif = "data:image/gif;base64,"+ gif_to_base64html(CUR_PATH, html_gif_name)
iris_imgif = iris_imgif+ """<img src="%s">""" % imsgif + """<br>"""

# method 2.2, this html page embeded one timely fresh script
root = "<title>Milize AI Predict Stock</title>"
root = root + iris_des + "<table>"+"</table>"  
#root = root + iris_des + "<table>"+ iris_im+"</table>"  
#if need ,you can append this content to below ,to refresh htm page.  +"""<script language="JavaScript"> setTimeout(function(){location.reload()},70000); </script>"""
root = "<h1> Milize AI Predict Animation </h1><br>"+iris_imgif  +"<br>"+ root
# lxml lib etree convert the info to html soure code ,and write it to file
html = etree.HTML(root)
tree = etree.ElementTree(html)
tree.write('C:/JINSHUXIN/Milize_AI/Milize_AI/templates/predict.html')
# open html file with explorer
webbrowser.open('C:/JINSHUXIN/Milize_AI/Milize_AI/templates/predict.html',new = 0.3)



#################################









