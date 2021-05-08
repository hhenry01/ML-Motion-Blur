from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# from . import tracking
import os
import pathlib
from pathlib import Path
from django.http import HttpResponse

# Create your views here.
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

def about(request):
    return render(request, 'main_site/about.html')

def processVid(request):
    parent_dir = pathlib.Path('views.py').parent.absolute()
    path = f'{parent_dir}/media/video'
    # model = tracking.load_model('Model.pt', len(tracking.classes) + 1)
    # detections, labels = tracking.track(model, path)

    return HttpResponse("<h1>Test page to process the video</h1>")