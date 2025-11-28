from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm
from .utils import process_image, generate_pdf
import os
from django.conf import settings

def index(request):
    """Render the upload page."""
    form = ImageUploadForm()
    return render(request, "index.html", {"form": form})

def process_image_view(request):
    """Handles image upload and processing."""
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES["image"]
            image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)

            # Save uploaded image
            with open(image_path, "wb") as f:
                for chunk in uploaded_image.chunks():
                    f.write(chunk)

            # Process image (count trees, crop, classify)
            results = process_image(image_path)

            # Generate PDF Report
            pdf_path = os.path.join(settings.MEDIA_ROOT, "tree_classification_report.pdf")
            generate_pdf(results, pdf_path)

            return render(request, "index.html", {
                "form": form,
                "results": results,
                "pdf_url": settings.MEDIA_URL + "tree_classification_report.pdf"
            })

    return render(request, "index.html", {"form": form})

def download_pdf(request):
    """Download the generated PDF report."""
    pdf_path = os.path.join(settings.MEDIA_ROOT, "tree_classification_report.pdf")
    with open(pdf_path, "rb") as pdf:
        response = HttpResponse(pdf.read(), content_type="application/pdf")
        response["Content-Disposition"] = 'attachment; filename="tree_classification_report.pdf"'
        return response
