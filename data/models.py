from django.db import models

class Ambiente(models.TextChoices):
    OPTIMO = 'óptimo', 'Óptimo'
    ACEPTABLE = 'aceptable', 'Aceptable'
    CRITICO = 'crítico', 'Crítico'
class MetricSpec(models.Model):
    # clave corta para usar en gráficos/joins con tus mediciones
    code = models.CharField(max_length=20, unique=True)  # CO2, RUIDO, TEMP, HUM, ENERG, TVOC
    sensor = models.CharField(max_length=80)             # p.ej. "CO₂", "Ruido / Sonido"
    measured_param = models.CharField(max_length=120)    # "Concentración de CO₂ en ambiente"
    input_desc = models.CharField(max_length=120)        # "ppm en tiempo real"
    unit = models.CharField(max_length=20)               # "ppm", "dB(A)", "%", "kWh"
    # Texto tal cual quieres mostrar en la leyenda/tabla del dashboard
    range_text = models.TextField()                      # "≤ 800 (mín) – 801–1200 (medio) – > 1200 (alto)"
    use_in_reports = models.CharField(max_length=200)    # "Calidad de aire interior, ..."

    # Umbrales numéricos para lógica (opcional pero útil)
    thresholds = models.JSONField(null=True, blank=True)
    # ejemplo: {"low_max":800,"mid_max":1200}  (alto = > mid_max)

    class Meta:
        verbose_name = "Especificación de parámetro"
        verbose_name_plural = "Especificaciones de parámetros"
        ordering = ["code"]

    def __str__(self):
        return f"{self.code} - {self.measured_param}"
    
class Measurement(models.Model):
    created_at = models.DateTimeField(db_index=True)
    temp_c = models.FloatField()
    co2_ppm = models.IntegerField()
    noise_dba = models.FloatField()
    humidity_pct = models.FloatField()
    ambiente = models.CharField(max_length=15, choices=Ambiente.choices)
    tvoc_index = models.IntegerField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = 'Medición'
        verbose_name_plural = 'Mediciones'
        constraints = [
            models.UniqueConstraint(
                fields=['created_at', 'temp_c', 'co2_ppm', 'noise_dba', 'humidity_pct'],
                name='uniq_measurement_at_values'
            )
        ]

    def __str__(self):
        return f"{self.created_at} | CO2 {self.co2_ppm} ppm | T {self.temp_c} °C"

