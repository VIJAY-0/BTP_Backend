from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response


# Create your views here.

from ImageProcessing import utils


@api_view(['POST'])
def process_image(request):
    # Example: Accessing POST data
    data = request.data
    image_file = request.FILES.get('image')
    print(data['image'])
    
    if not image_file:
        return Response({"error": "No image uploaded"}, status=400)
    
    data = utils.process_image(image_file)
    return Response({"message": "Image processed successfully",'data':data})


def image_processing_view(request):
    print(request)
    return HttpResponse("Image processing endpoint")
