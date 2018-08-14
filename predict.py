# -*- coding: utf-8 -*-
 
from django.http import HttpResponse
from django.shortcuts import render_to_response



# recieve the request data
def predict(request):  
    request.encoding='utf-8'
    #return HttpResponse('predict.html')
    return render_to_response('predict.html')
    #return render_to_response(('stock_predict.html'))

'''
# form 
def search_form(request):
    return render_to_response('search_form.html')
 
# recieve request form
def search(request):  
    request.encoding='utf-8'
    if 'q' in request.GET:
        message = 'You request: ' + request.GET['q']
    else:
        message = 'You submit blank request.'
        
        
    #how to response with actual chart result ????    
        
    return HttpResponse(message)
    

'''

