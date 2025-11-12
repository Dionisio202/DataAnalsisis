from datetime import datetime, timedelta
from django.db.models import Avg, Max, Min, Count, Sum, Case, When, IntegerField, Q
from django.db.models.functions import TruncHour, TruncDay, TruncDate, ExtractHour, ExtractIsoWeekDay, ExtractWeek , ExtractIsoYear
import math
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.dateparse import parse_date

from data.models import Measurement, MetricSpec, Ambiente

# Código -> (campo en Measurement, unidad default)
METRICS = {
    "CO2":   ("co2_ppm",      "ppm"),
    "RUIDO": ("noise_dba",    "dB(A)"),
    "TEMP":  ("temp_c",       "°C"),
    "HUM":   ("humidity_pct", "%"),
    "TVOC":  ("tvoc_index",   "índice"),
}


def _get_date_range(request):
    """
    Obtiene el rango de fechas desde los parámetros de la petición.
    Soporta tanto 'days' como 'start_date' y 'end_date'.
    
    Returns:
        tuple: (since_datetime, until_datetime)
    """
    start_date = request.GET.get("start_date")
    end_date = request.GET.get("end_date")
    
    if start_date and end_date:
        # Modo rango personalizado
        since = parse_date(start_date)
        until = parse_date(end_date)
        
        if since and until:
            # Convertir a datetime para incluir todo el día final
            since = datetime.combine(since, datetime.min.time())
            until = datetime.combine(until, datetime.max.time())
            return since, until
    
    # Modo días predefinidos (fallback)
    days = request.GET.get("days", "30")
    if days == "all":
        since = datetime.min.replace(year=1970)
        until = datetime.now()
    else:
        days = int(days)
        until = datetime.now()
        since = until - timedelta(days=days)
    
    return since, until


def _get_ambiente_filter(request):
    """
    Obtiene el filtro de ambiente desde los parámetros.
    Returns:
        str | None: El ambiente seleccionado o None para todos
    """
    ambiente = request.GET.get("ambiente")
    if ambiente and ambiente != "all":
        return ambiente
    return None


def _apply_filters(qs, since, until, ambiente=None):
    """
    Aplica filtros de fecha y ambiente a un queryset.
    """
    qs = qs.filter(created_at__gte=since, created_at__lte=until)
    if ambiente:
        qs = qs.filter(ambiente=ambiente)
    return qs


def _category_from_thresholds(code: str, value):
    if value is None:
        return {"key": "na", "label": "s/d", "color": "#9CA3AF"}

    spec = MetricSpec.objects.filter(code=code).first()
    th = (spec.thresholds if spec and spec.thresholds else {}) or {}

    low_max = th.get("low_max")
    mid_max = th.get("mid_max")
    low_min = th.get("low_min")

    if code == "HUM":
        low_min = low_min if low_min is not None else 40
        low_max = low_max if low_max is not None else 50
        mid_max = mid_max if mid_max is not None else 60
        if low_min <= value <= low_max:
            return {"key": "low", "label": "Óptimo", "color": "#10B981"}
        elif low_max < value <= mid_max:
            return {"key": "mid", "label": "Aceptable", "color": "#F59E0B"}
        else:
            return {"key": "high", "label": "Crítico", "color": "#EF4444"}

    low_max = low_max if low_max is not None else 800
    mid_max = mid_max if mid_max is not None else 1200
    if value <= low_max:
        return {"key": "low", "label": "Óptimo", "color": "#10B981"}
    elif value <= mid_max:
        return {"key": "mid", "label": "Aceptable", "color": "#F59E0B"}
    else:
        return {"key": "high", "label": "Crítico", "color": "#EF4444"}


def _metric_has_any_data(code: str) -> bool:
    field, _ = METRICS[code]
    return Measurement.objects.filter(**{f"{field}__isnull": False}).exists()


def _kpis_for_metric(code: str, since: datetime, until: datetime, ambiente=None):
    field, default_unit = METRICS[code]
    # Datos en el rango elegido
    qs_range = _apply_filters(Measurement.objects.all(), since, until, ambiente)
    agg = qs_range.aggregate(avg=Avg(field), min=Min(field), max=Max(field))

    # Último valor en el rango; si no hay, último global como fallback
    last = qs_range.order_by('-created_at').values_list(field, 'created_at').first()
    if last and last[0] is not None:
        last_value, last_at = last[0], last[1]
    else:
        last_global = (
            Measurement.objects
            .exclude(**{f"{field}__isnull": True})
            .order_by('-created_at')
            .values_list(field, 'created_at')
            .first()
        )
        last_value, last_at = (last_global[0], last_global[1]) if last_global else (None, None)

    spec = MetricSpec.objects.filter(code=code).first()
    unit = (spec.unit if spec else None) or default_unit

    return {
        "code": code,
        "unit": unit,
        "avg": agg["avg"],
        "min": agg["min"],
        "max": agg["max"],
        "last_value": last_value,
        "last_at": last_at,
        "category": _category_from_thresholds(code, last_value),
        "range_text": (spec.range_text if spec else ""),
        "label": (spec.measured_param if spec else code),
    }


def home(request):
    """
    Dashboard. El rango inicial es 30 días (puedes cambiarlo aquí).
    El frontend permite cambiarlo sin recargar la página.
    """
    default_days = 30
    now = datetime.now()
    since = now - timedelta(days=default_days)
    until = now

    # Sólo mostramos métricas con datos en la base
    order = ["CO2", "RUIDO", "TEMP", "HUM", "TVOC"]
    metrics = [m for m in order if _metric_has_any_data(m)]

    cards = [_kpis_for_metric(code, since, until) for code in metrics]

    # Pie de ambientes usando el mismo rango inicial
    ambiente_counts = (
        Measurement.objects
        .filter(created_at__gte=since, created_at__lte=until)
        .values('ambiente')
        .annotate(total=Count('id'))
    )
    ambiente_map = {a['ambiente']: a['total'] for a in ambiente_counts}
    ambiente_chart = {
        "labels": [label for _, label in Ambiente.choices],
        "data": [
            ambiente_map.get(Ambiente.OPTIMO, 0),
            ambiente_map.get(Ambiente.ACEPTABLE, 0),
            ambiente_map.get(Ambiente.CRITICO, 0),
        ],
    }

    specs = MetricSpec.objects.order_by('code')

    # Obtener opciones de ambiente para el filtro
    ambientes = [
        {"value": "all", "label": "Todos los ambientes"},
        {"value": Ambiente.OPTIMO, "label": "Óptimo"},
        {"value": Ambiente.ACEPTABLE, "label": "Aceptable"},
        {"value": Ambiente.CRITICO, "label": "Crítico"},
    ]

    context = {
        "metrics": metrics,
        "cards": cards,
        "ambiente_chart": ambiente_chart,
        "specs": specs,
        "initial_days": default_days,
        "ambientes": ambientes,
    }
    return render(request, "dashboard/index.html", context)


def api_summary(request):
    """
    API para obtener resumen de KPIs.
    Acepta: days=X o start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&ambiente=X
    """
    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)
    
    order = ["CO2", "RUIDO", "TEMP", "HUM", "TVOC"]
    metrics = [m for m in order if _metric_has_any_data(m)]
    cards = [_kpis_for_metric(code, since, until, ambiente) for code in metrics]
    return JsonResponse({"cards": cards})


def api_series(request, metric: str):
    """
    API para obtener series temporales de una métrica.
    Acepta: days=X o start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&ambiente=X
    """
    metric = metric.upper()
    if metric not in METRICS:
        return JsonResponse({"error": "metric not found"}, status=404)

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)
    
    field, unit = METRICS[metric]
    qs = _apply_filters(
        Measurement.objects.exclude(**{f"{field}__isnull": True}),
        since, 
        until, 
        ambiente
    )

    # Determinar granularidad según el rango
    delta = until - since
    
    if delta.days <= 2:
        # Para 2 días o menos, agrupar por hora
        grouped = (
            qs.annotate(t=TruncHour('created_at'))
              .values('t').order_by('t')
              .annotate(v=Avg(field))
        )
        labels = [x['t'].strftime("%d/%m %Hh") for x in grouped]
    elif delta.days <= 31:
        # Para un mes o menos, agrupar por día
        grouped = (
            qs.annotate(t=TruncDay('created_at'))
              .values('t').order_by('t')
              .annotate(v=Avg(field))
        )
        labels = [x['t'].strftime("%d/%m") for x in grouped]
    else:
        # Para rangos mayores, agrupar por día también
        grouped = (
            qs.annotate(t=TruncDay('created_at'))
              .values('t').order_by('t')
              .annotate(v=Avg(field))
        )
        labels = [x['t'].strftime("%d/%m/%y") for x in grouped]

    data = [x['v'] for x in grouped]
    
    # Obtener especificación para la unidad
    spec = MetricSpec.objects.filter(code=metric).first()
    if spec and spec.unit:
        unit = spec.unit
    
    return JsonResponse({
        "metric": metric,
        "unit": unit,
        "labels": labels,
        "data": data,
    })


def api_ambiente_pie(request):
    """
    API para obtener distribución por ambiente.
    Acepta: days=X o start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&ambiente=X
    """
    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)

    qs = _apply_filters(Measurement.objects.all(), since, until, ambiente)
    ambiente_counts = (
        qs.values('ambiente')
          .annotate(total=Count('id'))
    )
    ambiente_map = {a['ambiente']: a['total'] for a in ambiente_counts}
    return JsonResponse({
        "labels": [label for _, label in Ambiente.choices],
        "data": [
            ambiente_map.get(Ambiente.OPTIMO, 0),
            ambiente_map.get(Ambiente.ACEPTABLE, 0),
            ambiente_map.get(Ambiente.CRITICO, 0),
        ],
    })
# ---------- Helpers extra ----------
def _metric_field(metric_code: str):
    metric_code = metric_code.upper()
    if metric_code not in METRICS:
        return None, None
    return METRICS[metric_code]  # (field, unit)

def _thresholds(metric_code: str):
    spec = MetricSpec.objects.filter(code=metric_code).first()
    th = (spec.thresholds if spec and spec.thresholds else {}) or {}
    if metric_code == "HUM":
        low_min = th.get("low_min", 40)
        low_max = th.get("low_max", 50)
        mid_max = th.get("mid_max", 60)
        return low_max, mid_max, low_min
    else:
        low_max = th.get("low_max", 800)
        mid_max = th.get("mid_max", 1200)
        return low_max, mid_max, None
def _compute_scale(values, pad_pct=0.08, force_zero=False):
    """Calcula min/max 'apretados' para el eje a partir de la data."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    vmin, vmax = min(vals), max(vals)

    if force_zero:
        vmin = 0.0

    if vmin == vmax:
        # abre un margen cuando todo es igual
        delta = (abs(vmin) * 0.1) or 1.0
        vmin -= delta
        vmax += delta

    pad = (vmax - vmin) * pad_pct
    return {"min": round(vmin - pad, 4), "max": round(vmax + pad, 4)}


# ---------- 1) Días: óptimos/aceptables/críticos + día exacto y tipo de jornada ----------
def api_daily_categories(request):
    """
    Por día:
      - conteo de óptimos / aceptables / críticos (según thresholds)
      - weekday (1=lunes ... 7=domingo), is_weekend
    Además:
      - summary: total_days, days_ok (mayoría óptimo por día)
      - top_critical_days: top 10 días con más críticos
    Params: metric=CO2|RUIDO|TEMP|HUM|TVOC + (days|start_date/end_date) + ambiente
            normalize=1 para devolver porcentajes 0..100 (barras 100% apiladas)
    """
    metric = (request.GET.get("metric") or "CO2").upper()
    field, _ = _metric_field(metric)
    if not field:
        return JsonResponse({"error": "metric not found"}, status=404)

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)
    low_max, mid_max, low_min = _thresholds(metric)

    base = _apply_filters(
        Measurement.objects.exclude(**{f"{field}__isnull": True}),
        since, until, ambiente
    ).annotate(
        d=TruncDate('created_at'),
        wd=ExtractIsoWeekDay('created_at'),   # 1..7
    )

    if metric == "HUM":
        cond_opt = Q(**{f"{field}__gte": low_min}) & Q(**{f"{field}__lte": low_max})
        cond_acc = Q(**{f"{field}__gt": low_max}) & Q(**{f"{field}__lte": mid_max})
        cond_cri = ~Q(**{f"{field}__gte": low_min}) | ~Q(**{f"{field}__lte": mid_max})
    else:
        cond_opt = Q(**{f"{field}__lte": low_max})
        cond_acc = Q(**{f"{field}__gt": low_max}) & Q(**{f"{field}__lte": mid_max})
        cond_cri = Q(**{f"{field}__gt": mid_max})

    daily = (base.values('d', 'wd')
        .annotate(
            optimos=Sum(Case(When(cond_opt, then=1), default=0, output_field=IntegerField())),
            aceptables=Sum(Case(When(cond_acc, then=1), default=0, output_field=IntegerField())),
            criticos=Sum(Case(When(cond_cri, then=1), default=0, output_field=IntegerField())),
            total=Count('id'),
        )
        .order_by('d'))

    normalize = (request.GET.get("normalize") == "1")

    rows, days_ok = [], 0
    totals_for_scale = []

    for x in daily:
        total = x['total'] or 1
        o, a, c = x['optimos'], x['aceptables'], x['criticos']
        majority_opt = o > (total / 2.0)
        if majority_opt:
            days_ok += 1

        if normalize:
            o = 100.0 * o / total
            a = 100.0 * a / total
            c = 100.0 * c / total
        else:
            totals_for_scale.append(o + a + c)

        rows.append({
            "date": x['d'].isoformat(),
            "weekday": int(x['wd']),                 # 1..7
            "is_weekend": int(x['wd']) in (6, 7),
            "optimos": o,
            "aceptables": a,
            "criticos": c,
            "total": x['total'],
        })

    top_crit = sorted(rows, key=lambda r: r['criticos'], reverse=True)[:10]

    payload = {
        "metric": metric,
        "rows": rows,
        "summary": {"total_days": len(rows), "days_ok": days_ok},
        "top_critical_days": top_crit,
        "normalized": normalize,
    }

    # Si no normalizas, devolvemos escala apretada para el eje de totales apilados
    if not normalize:
        payload["scale"] = _compute_scale(totals_for_scale, pad_pct=0.05, force_zero=True)

    return JsonResponse(payload)


# ---------- 2) Agregados por tipo de día (laborable/fin de semana/domingo) o weekday específico ----------
def api_by_daytype(request):
    """
    Promedios diarios según tipo de día.
    Params:
      metric=...,
      daytype=all|laborable|weekend|domingo,
      weekday=1..7 (opcional; si se envía, tiene prioridad),
      + rango/ambiente
    """
    metric = (request.GET.get("metric") or "RUIDO").upper()
    field, unit = _metric_field(metric)
    if not field:
        return JsonResponse({"error":"metric not found"}, status=404)

    daytype = (request.GET.get("daytype") or "all").lower()
    weekday = request.GET.get("weekday")

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)

    base = _apply_filters(
        Measurement.objects.exclude(**{f"{field}__isnull": True}),
        since, until, ambiente
    ).annotate(
        d=TruncDate('created_at'),
        wd=ExtractIsoWeekDay('created_at'),
    )

    if weekday:
        base = base.filter(wd=int(weekday))
    else:
        if daytype == "laborable":
            base = base.filter(wd__in=[1,2,3,4,5])
        elif daytype == "weekend":
            base = base.filter(wd__in=[6,7])
        elif daytype == "domingo":
            base = base.filter(wd=7)

    per_day = (base.values('d')
        .annotate(avg=Avg(field), min=Min(field), max=Max(field), n=Count('id'))
        .order_by('d'))

    rows = [{
        "date": x['d'].isoformat(),
        "avg": x['avg'], "min": x['min'], "max": x['max'], "n": x['n']
    } for x in per_day]

    scale = _compute_scale([r["avg"] for r in rows], pad_pct=0.08, force_zero=False)

    return JsonResponse({"metric": metric, "unit": unit, "rows": rows, "scale": scale})

# ---------- 3) Horas pico (global) ----------
def api_peak_hours(request):
    """
    Histograma por hora del día y top horas por promedio.
    Params: metric=..., top=5 + rango/ambiente
    """
    metric = (request.GET.get("metric") or "CO2").upper()
    field, _ = _metric_field(metric)
    if not field:
        return JsonResponse({"error":"metric not found"}, status=404)

    top = int(request.GET.get("top", "5"))
    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)

    qs = _apply_filters(
        Measurement.objects.exclude(**{f"{field}__isnull": True}),
        since, until, ambiente
    ).annotate(h=ExtractHour('created_at'))

    hours = (qs.values('h')
        .annotate(avg=Avg(field), max=Max(field), n=Count('id'))
        .order_by('h'))

    rows = [{"hour": int(x['h']), "avg": x['avg'], "max": x['max'], "n": x['n']} for x in hours]
    top_hours = sorted(rows, key=lambda r: (r['avg'] if r['avg'] is not None else -1), reverse=True)[:top]

    scale = _compute_scale([r["avg"] for r in rows], pad_pct=0.10, force_zero=False)

    return JsonResponse({"metric": metric, "rows": rows, "top_hours": top_hours, "scale": scale})


# ---------- 4) Hora pico por semana ----------
def api_weekly_peak_hour(request):
    """
    Para cada (año ISO, semana ISO) devuelve la hora con mayor promedio.
    Params: metric=... + rango/ambiente
    """
    metric = (request.GET.get("metric") or "CO2").upper()
    field, _ = _metric_field(metric)
    if not field:
        return JsonResponse({"error":"metric not found"}, status=404)

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)

    qs = _apply_filters(
        Measurement.objects.exclude(**{f"{field}__isnull": True}),
        since, until, ambiente
    ).annotate(
        y=ExtractIsoYear('created_at'),
        w=ExtractWeek('created_at'),
        h=ExtractHour('created_at'),
    )

    by_ywh = qs.values('y', 'w', 'h').annotate(avg=Avg(field)).order_by('y', 'w', 'h')

    best = {}
    for x in by_ywh:
        key = (int(x['y']), int(x['w']))
        cand = best.get(key)
        if not cand or (x['avg'] or -math.inf) > (cand['avg'] or -math.inf):
            best[key] = {
                "year": int(x['y']),
                "week": int(x['w']),
                "hour": int(x['h']),
                "avg": x['avg'],
            }

    rows = [best[k] for k in sorted(best.keys())]
    return JsonResponse({"metric": metric, "rows": rows})


# ---------- 5) Comparación de variables + correlación (Pearson) ----------
def api_compare_variables(request):
    """
    Series alineadas por hora de dos variables y su correlación.
    Params: x=CO2, y=TEMP + rango/ambiente
    """
    x = (request.GET.get("x") or "CO2").upper()
    y = (request.GET.get("y") or "TEMP").upper()
    xf, _ = _metric_field(x)
    yf, _ = _metric_field(y)
    if not xf or not yf or x == y:
        return JsonResponse({"error":"invalid variables"}, status=400)

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)

    base = _apply_filters(Measurement.objects.all(), since, until, ambiente).annotate(t=TruncHour('created_at'))
    X = base.values('t').annotate(v=Avg(xf))
    Y = base.values('t').annotate(v=Avg(yf))

    mapX = {r['t']: r['v'] for r in X}
    mapY = {r['t']: r['v'] for r in Y}
    ts = sorted(set(mapX.keys()) & set(mapY.keys()))

    series = [{"t": t.isoformat(), x: mapX[t], y: mapY[t]} for t in ts]

    pairs = [(mapX[t], mapY[t]) for t in ts if mapX[t] is not None and mapY[t] is not None]
    if len(pairs) >= 3:
        ax = [p[0] for p in pairs]; ay = [p[1] for p in pairs]
        mx = sum(ax)/len(ax); my = sum(ay)/len(ay)
        num = sum((ax[i]-mx)*(ay[i]-my) for i in range(len(ax)))
        denx = sum((ax[i]-mx)**2 for i in range(len(ax))) ** 0.5
        deny = sum((ay[i]-my)**2 for i in range(len(ay))) ** 0.5
        corr = (num/(denx*deny)) if denx and deny else 0.0
    else:
        corr = None

    return JsonResponse({"x": x, "y": y, "series": series, "pearson": corr})

# ---------- 6) Cruce de picos: qué pasa en B cuando A tiene pico ----------
def api_cross_peaks(request):
    """
    Toma picos en A (> mid_max) y promedia B en ventana alrededor.
    Params: a=CO2, b=RUIDO, window_min=30, max_peaks=100 + rango/ambiente
    """
    a = (request.GET.get("a") or "CO2").upper()
    b = (request.GET.get("b") or "RUIDO").upper()
    af, _ = _metric_field(a)
    bf, _ = _metric_field(b)
    if not af or not bf:
        return JsonResponse({"error":"invalid variables"}, status=400)

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)
    _, mid_max, _ = _thresholds(a)

    base = _apply_filters(Measurement.objects.all(), since, until, ambiente)
    peaks = list(base.filter(**{f"{af}__gt": mid_max}).order_by('created_at')
                      .values_list('created_at', flat=True))

    win = int(request.GET.get("window_min", "30"))
    max_peaks = int(request.GET.get("max_peaks", "100"))
    if max_peaks > 0:
        peaks = peaks[:max_peaks]

    out = []
    for t in peaks:
        lo = t - timedelta(minutes=win)
        hi = t + timedelta(minutes=win)
        avg_b = (base.filter(created_at__gte=lo, created_at__lte=hi)
                      .aggregate(v=Avg(bf)))['v']
        out.append({"t": t.isoformat(), "avg_b_window": avg_b})

    return JsonResponse({"a": a, "b": b, "window_min": win, "rows": out})

# ---------- 7) Regresión multivariante: CO2 ~ (TEMP, HUM, RUIDO, TVOC) ----------
def api_regression(request):
    """
    Regresión lineal múltiple: target (default CO2) ~ features.
    Params:
      target=CO2
      features=TEMP,HUM,RUIDO,TVOC (lista, sin incluir target)
      freq=hour|day
      + rango/ambiente
    Devuelve coeficientes y R^2.
    """
    target = (request.GET.get("target") or "CO2").upper()
    feats = (request.GET.get("features") or "TEMP,HUM,RUIDO").upper().split(",")
    freq = (request.GET.get("freq") or "hour").lower()

    tf, _ = _metric_field(target)
    f_fields = []
    for f in feats:
        ff, _ = _metric_field(f)
        if ff and f != target:
            f_fields.append((f, ff))
    if not tf or not f_fields:
        return JsonResponse({"error":"invalid target/features"}, status=400)

    since, until = _get_date_range(request)
    ambiente = _get_ambiente_filter(request)

    base = _apply_filters(Measurement.objects.all(), since, until, ambiente)
    base = base.annotate(t=TruncDay('created_at') if freq == "day" else TruncHour('created_at'))

    agg = base.values('t').annotate(
        y=Avg(tf),
        **{f"f_{name}": Avg(ff) for (name, ff) in f_fields}
    ).order_by('t')

    X, y = [], []
    for r in agg:
        if r['y'] is None:
            continue
        row = []
        ok = True
        for (name, _ff) in f_fields:
            v = r.get(f"f_{name}")
            if v is None:
                ok = False; break
            row.append(float(v))
        if ok:
            X.append([1.0] + row)  # intercepto
            y.append(float(r['y']))

    if len(X) < len(f_fields) + 2:
        return JsonResponse({"error":"not enough data"}, status=400)

    # Normal equation
    try:
        import numpy as np
    except ImportError:
        return JsonResponse({"error":"numpy not installed on server"}, status=500)

    X = np.array(X)  # n x (k+1)
    y = np.array(y)  # n
    XtX = X.T @ X
    try:
        beta = np.linalg.inv(XtX) @ X.T @ y
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ X.T @ y

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0

    coefs = {"intercept": float(beta[0])}
    for i, (name, _ff) in enumerate(f_fields, start=1):
        coefs[name] = float(beta[i])

    return JsonResponse({
        "target": target,
        "features": [name for (name, _ff) in f_fields],
        "freq": freq,
        "n": int(X.shape[0]),
        "r2": r2,
        "coefficients": coefs
    })