from django.core.management.base import BaseCommand
from data.models import MetricSpec

DATA = [
    {
        "code": "CO2",
        "sensor": "CO₂",
        "measured_param": "Concentración de CO₂ en ambiente",
        "input_desc": "ppm en tiempo real",
        "unit": "ppm",
        "range_text": "≤ 800 ppm (mín) – 801–1200 ppm (medio) – > 1200 ppm (alto)",
        "use_in_reports": "Calidad de aire interior, cumplimiento de normativas (ASHRAE/OMS)",
        "thresholds": {"low_max": 800, "mid_max": 1200},
    },
    {
        "code": "RUIDO",
        "sensor": "Ruido / Sonido",
        "measured_param": "Nivel de presión sonora",
        "input_desc": "Nivel dB(A) medido en continuo",
        "unit": "dB(A)",
        "range_text": "≤ 70 dB (mín) – 71–85 dB (medio) – > 85 dB (alto)",
        "use_in_reports": "Salud ocupacional, seguridad industrial",
        "thresholds": {"low_max": 70, "mid_max": 85},
    },
    {
        "code": "TEMP",
        "sensor": "Temperatura",
        "measured_param": "Temperatura ambiente",
        "input_desc": "°C por intervalo",
        "unit": "°C",
        "range_text": "18–22 °C (mín) – 23–26 °C (medio) – > 26 °C (alto)",
        "use_in_reports": "Confort térmico, eficiencia HVAC",
        "thresholds": {"low_max": 22, "mid_max": 26},
    },
    {
        "code": "HUM",
        "sensor": "Humedad relativa",
        "measured_param": "Humedad ambiental",
        "input_desc": "%HR por intervalo",
        "unit": "%",
        "range_text": "40–50% (mín) – 51–60% (medio) – < 40% o > 60% (alto)",
        "use_in_reports": "Confort ambiental, prevención de moho",
        "thresholds": {"low_min": 40, "low_max": 50, "mid_max": 60},
    },
    {
        "code": "ENERG",
        "sensor": "Energía eléctrica",
        "measured_param": "Consumo de energía por área/equipo",
        "input_desc": "kWh (periodo)",
        "unit": "kWh",
        "range_text": "ICE ≤ 6 kWh/m²·mes (mín) – 6,1–10 kWh/m²·mes (medio) – > 10 kWh/m²·mes (alto)",
        "use_in_reports": "Eficiencia energética, cálculo de huella de carbono (tCO₂e)",
        "thresholds": {"low_max": 6.0, "mid_max": 10.0},
    },
    {
        "code": "TVOC",
        "sensor": "TVOC",
        "measured_param": "Índice de compuestos orgánicos volátiles",
        "input_desc": "índice en tiempo real",
        "unit": "índice",
        "range_text": "≤ 220 (mín) – 221–660 (medio) – > 660 (alto) (ajusta a tu estándar)",
        "use_in_reports": "Calidad de aire interior",
        "thresholds": {"low_max": 220, "mid_max": 660},
    },
]

class Command(BaseCommand):
    help = "Carga/actualiza especificaciones de parámetros para dashboards y reportes."

    def handle(self, *args, **kwargs):
        for item in DATA:
            MetricSpec.objects.update_or_create(code=item["code"], defaults=item)
        self.stdout.write(self.style.SUCCESS("MetricSpec sembrado correctamente."))
