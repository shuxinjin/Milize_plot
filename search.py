# -*- coding: utf-8 -*-
import sys
sys.path.append('c:/JINSHUXIN/Milize_AI/Milize_AI/')


from django.http import HttpResponse
from django.shortcuts import render_to_response

# I means you can choose the module to import, or as users request to import.
# Keep for extend design and coding 

import_flag=1

if import_flag:
    from Milize_AI import milizeshowpic6 as milizeshow
else:
    pass
# The core module to draw a AI chart,and show it to customers


'''
#user for redirect
from flask import  Flask
from flask import  make_response
from flask import redirect
from flask import abort
'''
 

def search_form(request):
    return render_to_response('search_form.html')


# recieve the request data
def search(request):  
    request.encoding='utf-8'
    ainame=""
    passvalue=""
    ##return HttpResponse('stock_predict.html')
    if 'q' in request.GET:
        message = 'You request: ' + request.GET['q']
        
        # keep here for future to get data from client 
        ainame = request.GET['ainame'] 
        passvalue = request.GET['passvalue']
        
        #execute client main request 
        do_ai_info(ainame, passvalue)
        return render_to_response(('predict.html'))
    else:
        message = 'Blank request'
        return HttpResponse(message)
    
  


def do_ai_info(ainame, passvalue):   
  try:
      aa="ai algorithm name :"+ainame +" User submit  value :"+passvalue
      #+"ai algorithm name :"+ainame +" User submit  value :"+passvalue
      #aa="Predict kernal procedure write by milize,Be called :"+milizeshow.ttest
      print(aa)
      return HttpResponse(aa)
      
    # redirect
    #@app.route('/redirect')
    
    ###redirect("http://192.168.1.6:8000/predict")
    

    #conn = sqlite3.connect(db_path) 
    #sql = "insert into t_user(username, password) values('%s', '%s')" % (username, password) 
    #conn.execute(sql) 
    #conn.commit() 
    #conn.close() 
    
    
    
    #print (milizeshow.test())
    #return HttpResponse(milizeshow.test())
    

  except Exception: 
    print ('------', str(Exception))
    return HttpResponse("in do_ai_info.except .")
    '''
    try: 
      #conn.close() 
    except Exception:
        
      pass
    '''


