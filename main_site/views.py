from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from torch import nn, optim
from torchvision import datasets, models, transforms

# Create your views here.
def home(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(name)
        context['url'] = fs.url(name)

    return render(request, 'main_site/home.html', context)

def about(request):
    return render(request, 'main_site/about.html')

def processVid(request):
    model=torch.load('Model.pt', map_location=torch.device('cpu'))
    model.eval()