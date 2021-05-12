from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from . import tracking
from . import motion_blurrer
import os
import pathlib
from pathlib import Path
from django.http import HttpResponse
import torch

# Create your views here.
# Main method that returns the home screen, being able to take in the video and store it
def home(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save('video', uploaded_file)
        url = fs.url(name)
        context['url'] = fs.url(name)
        return processVid(request)
    
    return render(request, 'main_site/home.html', context)

# Method to return a little context on the project
def about(request):
    return render(request, 'main_site/about.html')

# Method to process the video using Henry and William functions
def processVid(request):
    context = {}
    parent_dir = pathlib.Path('views.py').parent.absolute()
    path = f'{parent_dir}/media/video'
    context['path'] = f'{parent_dir}/media/video'
    model = tracking.load_model(f'{parent_dir}/main_site/Model.pt', 17)
    boxes, ids = tracking.track(model, path, torch.device('cpu'))
    motion_blur.controller(boxes, ids, path, f'{parent_dir}/main_site/frames', f'{parent_dir}/media')

    return render(request, 'main_site/processVid.html', context)