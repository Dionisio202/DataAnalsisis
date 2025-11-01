from django.contrib import admin, messages
from django.urls import path
from django.shortcuts import render, redirect
from django import forms

from .models import Measurement, MetricSpec
from .services.import_measurements import import_measurements_file

class UploadFileForm(forms.Form):
    file = forms.FileField(label="Archivo (.csv o .xlsx)")

@admin.register(Measurement)
class MeasurementAdmin(admin.ModelAdmin):
    list_display = ("created_at", "co2_ppm", "noise_dba", "temp_c", "humidity_pct", "ambiente")
    list_filter = ("ambiente",)
    search_fields = ("created_at",)
    date_hierarchy = "created_at"
    change_list_template = "admin/data/measurement_changelist.html"  # añade botón

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path("upload/", self.admin_site.admin_view(self.upload_view), name="data_measurement_upload"),
        ]
        return my_urls + urls

    def upload_view(self, request):
        if request.method == "POST":
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                try:
                    created, errors = import_measurements_file(form.cleaned_data['file'])
                    messages.success(request, f"Insertadas: {created} | Errores: {len(errors)}")
                    if errors:
                        messages.warning(request, "Muestra de errores:\n" + "\n".join(errors[:5]))
                    return redirect("..")
                except Exception as e:
                    messages.error(request, str(e))
        else:
            form = UploadFileForm()
        return render(request, "admin/data/upload_form.html", {"form": form})

@admin.register(MetricSpec)
class MetricSpecAdmin(admin.ModelAdmin):
    list_display = ("code", "sensor", "unit")
    search_fields = ("code", "sensor", "measured_param")
