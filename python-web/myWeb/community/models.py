from django.db import models

# Create your models here.
class Article(models.Model):
  name = models.CharField(max_length=50)
  title = models.CharField(max_length=50)
  contents = models.TextField()
  email = models.URLField()
  cdate = models.DateTimeField(auto_now_add=True)
