from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from .services.import_measurements import import_measurements_file

@csrf_exempt
def upload_measurements(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return HttpResponseBadRequest("Envía un archivo con clave 'file' por POST.")
    try:
        created, errors = import_measurements_file(request.FILES['file'])
        return JsonResponse({"inserted": created, "errors": errors[:50], "errors_count": len(errors)})
    except Exception as e:
        return HttpResponseBadRequest(str(e))
