#!C:\Python\Python38\python.exe
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.http import Http404, HttpResponseNotFound

# Create your views here.


def index(request):
  msg = 'My Message'
  #return HttpResponse("Hello, World!")
  #return HttpResponse("<h1>Hello, Django!</h1>")
  return render(request, 'index.html', {'message': msg})

def error(request):
  #return HttpResponseNotFound('<h1>not found</h1>')
  raise Http404("Not Found")  
