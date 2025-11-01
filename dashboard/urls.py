from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/summary/', views.api_summary, name='api_summary'),
    path('api/series/<str:metric>/', views.api_series, name='api_series'),
    path('api/ambiente-pie/', views.api_ambiente_pie, name='api_ambiente_pie'),  # <-- NUEVO

    # NUEVOS
    path('api/daily-categories/', views.api_daily_categories, name='api_daily_categories'),
    path('api/by-daytype/', views.api_by_daytype, name='api_by_daytype'),
    path('api/peak-hours/', views.api_peak_hours, name='api_peak_hours'),
    path('api/weekly-peak-hour/', views.api_weekly_peak_hour, name='api_weekly_peak_hour'),
    path('api/compare/', views.api_compare_variables, name='api_compare_variables'),
    path('api/cross-peaks/', views.api_cross_peaks, name='api_cross_peaks'),
    path('api/regression/', views.api_regression, name='api_regression'),
]
