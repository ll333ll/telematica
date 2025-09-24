import subprocess
from pathlib import Path
from datetime import datetime
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, Tuple

# Importar funciones de simulación para calcular métricas numéricas
from simulation import run_simulation


ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def ensure_figures():
    # 4 escenarios base, sin TCPCustom
    run(["python3", "simulation.py", "--all-scenarios", "--sim-time", "100", "--reno", "3", "--cubic", "3", "--bbr", "3", "--custom", "0"]) 
    # Parte 3: comparar con TCPCustom en 100/30
    run(["python3", "simulation.py", "--capacity", "100", "--buffer", "30", "--sim-time", "100", "--reno", "3", "--cubic", "3", "--bbr", "3", "--custom", "3"]) 


def wrap_text_to_axes(ax, text, title=None):
    ax.axis('off')
    if title:
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, fontweight='bold')
    wrapped = textwrap.fill(text, width=110)
    ax.text(0.05, 0.88 if title else 0.95, wrapped, ha='left', va='top', fontsize=10)


def add_image_page(pdf, img_path, title=None):
    fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    if title:
        ax.text(0.5, 0.97, title, ha='center', va='top', fontsize=12, fontweight='bold')
    img = plt.imread(str(img_path))
    ax.imshow(img)
    pdf.savefig(fig)
    plt.close(fig)


def compile_pdf():
    out_pdf = ROOT / 'report.pdf'
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Cargar texto base del reporte (si existe)
    report_md = (ROOT / 'report.md').read_text(encoding='utf-8') if (ROOT / 'report.md').exists() else ''

    with PdfPages(out_pdf) as pdf:
        # Portada
        fig = plt.figure(figsize=(8.3, 11.7))  # A4 portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.text(0.5, 0.8, 'Simulación de control de congestión de TCP', ha='center', fontsize=18, fontweight='bold')
        ax.text(0.5, 0.74, 'Enlace compartido — Actividad opcional', ha='center', fontsize=12)
        ax.text(0.5, 0.68, now, ha='center', fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Resumen/introducción desde report.md
        if report_md:
            intro = report_md.split('##', 1)[0]
            fig = plt.figure(figsize=(8.3, 11.7))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            wrap_text_to_axes(ax, intro, title='Resumen')
            pdf.savefig(fig); plt.close(fig)

        # Definiciones formales (Throughput, RTT, Jain, Descartes)
        defs = (
            "Definiciones formales\n\n"
            "Throughput (por cliente i y paso t): x_i(t) = paquetes servidos en t.\n"
            "Throughput promedio del cliente i: X_i = (1/T) * ∑_{t=1..T} x_i(t).\n\n"
            "RTT (Round-Trip Time): tiempo ida y vuelta. En esta simulación,\n"
            "modelado como RTT(t) = RTT_base + α * (OcupaciónCola(t)), donde\n"
            "OcupaciónCola(t) = len(buffer)/BUFFER_SIZE y α = 50 ms.\n\n"
            "Índice de Jain (fairness) en t: J(t) = ( (∑_i x_i(t))^2 ) / ( n * ∑_i x_i(t)^2 ).\n"
            "Interpretación: J ∈ (0,1]; 1 implica reparto perfectamente equitativo;\n"
            "valores menores indican inequidad entre flujos.\n\n"
            "Descartes por algoritmo: suma de paquetes rechazados por overflow del\n"
            "búfer. Captura cuán agresivo es un conjunto de flujos bajo un escenario\n"
            "(o cuánta presión ejercen sobre el enlace).\n\n"
            "Nota: el objetivo de estas métricas es comparar CUBIC/BBR/Reno\n"
            "en términos de utilización, latencia (bufferbloat), equidad y presión\n"
            "sobre la cola."
        )
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        wrap_text_to_axes(ax, defs, title='Definiciones')
        pdf.savefig(fig); plt.close(fig)

        # Escenarios base
        scenarios = [(10, 10), (10, 30), (100, 10), (100, 30)]
        for cap, buf in scenarios:
            suffix = f"cap{cap}_buf{buf}"
            base_title = f"Escenario: Capacidad={cap} pps, Búfer={buf} pkts"
            # Panel principal
            panel = ROOT / f"congestion_simulation_results_with_custom_{suffix}.png"
            if panel.exists():
                add_image_page(pdf, panel, title=base_title + ' — Panel')
            # Fairness
            fair = ROOT / f"fairness_jain_{suffix}.png"
            if fair.exists():
                add_image_page(pdf, fair, title=base_title + ' — Fairness (Jain)')
            # Drops
            drops = ROOT / f"drops_by_algorithm_{suffix}.png"
            if drops.exists():
                add_image_page(pdf, drops, title=base_title + ' — Descartes por algoritmo')

        # Parte 3: escenario con TCPCustom (100/30)
        suffix = "cap100_buf30"
        custom_title = "Parte 3: Comparación con TCPCustom (Cap=100, Búfer=30)"
        panel = ROOT / f"congestion_simulation_results_with_custom_{suffix}.png"
        if panel.exists():
            add_image_page(pdf, panel, title=custom_title + ' — Panel')
        fair = ROOT / f"fairness_jain_{suffix}.png"
        if fair.exists():
            add_image_page(pdf, fair, title=custom_title + ' — Fairness (Jain)')
        drops = ROOT / f"drops_by_algorithm_{suffix}.png"
        if drops.exists():
            add_image_page(pdf, drops, title=custom_title + ' — Descartes por algoritmo')

        # Resumen numérico (Jain promedio y descartes por algoritmo)
        def jain(values):
            s = sum(values)
            s2 = sum(v*v for v in values)
            n = len(values)
            return (s*s) / (n * s2) if s2 > 0 and n > 0 else 0.0

        def summarize(cap: int, buf: int, custom: int) -> Tuple[float, Dict[str, int]]:
            hist = run_simulation(sim_time=100, capacity=cap, buffer_size=buf,
                                  n_reno=3, n_cubic=3, n_bbr=3, n_custom=custom, seed=42)
            th_series = [data["throughput"] for _, data in hist.items()]
            fairness_over_time = [jain([series[t] for series in th_series]) for t in range(len(th_series[0]))]
            jain_avg = sum(fairness_over_time) / len(fairness_over_time)
            drops_by_algo = {"Reno": 0, "Cubic": 0, "BBR": 0, "Custom": 0}
            for cid, data in hist.items():
                total_drops = sum(data.get("drops", []))
                for k in list(drops_by_algo.keys()):
                    if k in cid:
                        drops_by_algo[k] += total_drops
                        break
            return jain_avg, drops_by_algo

        # página de resumen
        fig = plt.figure(figsize=(8.3, 11.7))
        ax = fig.add_axes([0.08, 0.06, 0.84, 0.9])
        ax.axis('off')
        ax.text(0.5, 0.97, 'Resumen numérico (Jain promedio y descartes)', ha='center', fontsize=14, fontweight='bold')
        y = 0.9
        for cap, buf, custom in [(10,10,0),(10,30,0),(100,10,0),(100,30,0),(100,30,3)]:
            javg, drops = summarize(cap, buf, custom)
            title = f"Escenario Cap={cap} pps, Buffer={buf} pkts" + (" (+Custom)" if custom else "")
            line1 = f"• Jain promedio: {javg:.3f}"
            dd = ", ".join([f"{k}: {v}" for k, v in drops.items() if v > 0]) or "Sin descartes"
            line2 = f"• Descartes por algoritmo: {dd}"
            ax.text(0.02, y, title, fontsize=12, fontweight='bold'); y -= 0.05
            ax.text(0.04, y, line1, fontsize=11); y -= 0.04
            ax.text(0.04, y, line2, fontsize=11); y -= 0.06
        pdf.savefig(fig); plt.close(fig)

        # Conclusión desde report.md (última sección)
        if report_md:
            # toma contenido a partir de '## 6. Conclusión' si existe
            if '## 6. Conclusión' in report_md:
                concl = report_md.split('## 6. Conclusión', 1)[1]
                concl = 'Conclusión' + concl
            else:
                concl = 'Conclusión\n' + report_md[-1200:]
            fig = plt.figure(figsize=(8.3, 11.7))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            wrap_text_to_axes(ax, concl)
            pdf.savefig(fig); plt.close(fig)

    print(f"PDF generado en {out_pdf}")


def main():
    ensure_figures()
    compile_pdf()


if __name__ == '__main__':
    main()
