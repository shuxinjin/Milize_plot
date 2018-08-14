# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:19:52 2018

@author: sxjin
"""

from django.http import HttpResponse
from django.shortcuts import render



def hello(request):
    context           ={}
    context['hello']  ='Perform Milize AI Stock Predict ,main page ,welcome you .'
    return render(request,'hello.html',context)
    #return HttpResponse("Hello world ! ")




    
'''    

 
def hello(request):
    return HttpResponse("Hello world ! ")
    
    



 




    
'''