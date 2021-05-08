from django.db import models
from .validators import validate_file_extension

# Create your models here.
# class Video(models.Model):
#     title = models.CharField(max_length=100)
#     url = models.FileField(upload_to='videos/urls/')

#     def __str__(self):
#         return self.title