"""Milize_AI URL Configuration
#   该 Django 项目的 URL 声明; 一份由 Django 驱动的网站"目录"。
The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

"""
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
"""





from django.conf.urls import url
 
from . import view,search,predict,datainput
 
urlpatterns = [
    url(r'^$', view.hello),
    url(r'^datainput$', datainput.datainput),
    url(r'^search-form$', search.search_form),
    url(r'^search$', search.search),
    url(r'^predict$', predict.predict),
    #url(r'^ttest$', ttest.ttest),
]


'''
from django.conf.urls import url
 
from . import view
 
urlpatterns = [
    url(r'^$', view.hello),
]




from django.conf.urls import url
 
from . import view,search
 
urlpatterns = [
    url(r'^hello$', view.hello),
    url(r'^search-form$', search.search_form),
    url(r'^search$', search.search),
]






from . import view,testdb,search
 
urlpatterns = [
    url(r'^hello$', view.hello),
    url(r'^testdb$', testdb.testdb),
    url(r'^search-form$', search.search_form),
    url(r'^search$', search.search),
    
'''