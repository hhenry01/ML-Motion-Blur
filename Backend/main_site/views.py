from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Create your views here.
def home(request):
    context = {}
    if request.method == 'POST':
        # uploaded_file = request.FILES['video']
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(name)
        context['url'] = fs.url(name)
    return render(request, 'main_site/home.html', context)

# NEW
# def video_list(request):
#     return render(request, 'video_list.html')

# def upload_video(request):
#     return render(request, 'upload_video.html')