from django import forms

class UploadFileForm(forms.Form):
    dataset = forms.CharField(max_length=100)
    algorithm = forms.CharField(max_length=100)
