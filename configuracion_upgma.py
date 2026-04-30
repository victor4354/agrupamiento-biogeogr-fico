# configuracion_upgma.py
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from upgma_compacto_modular import pipeline, cortar_dendrograma
except ImportError:
    print("ERROR: No se encontró el archivo 'upgma_compacto_modular.py'.")
    print("       Asegúrate de que esté en la misma carpeta que este script.")
    sys.exit(1)


# ─────────────────────────────────────────────
#  CONFIGURACIÓN — edita aquí tus parámetros
# ─────────────────────────────────────────────

def cargar_configuracion():
    """
    Parámetros para el análisis de similitud y clustering jerárquico.
    Modificar según el estudio.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    configuracion = {
        # Archivo de entrada (Excel)
        'input_path': 'datos.xlsx',
        'sheet': None,                  # Nombre o índice de hoja (None = primera)

        # Columnas (None = autodetección)
        'species_col': None,            # Columna de especies/taxones
        'state_col': None,              # Columna de estados/regiones

        # Parámetros de análisis
        'sim_index': 'jaccard',         # jaccard, simpson, sorensen, ochiai,
                                        # braun-blanquet, fager, kulezynski,
                                        # correlation, baroni
        'linkage_method': 'average',    # single, complete, average (UPGMA)
        'min_species': 3,               # Filtrar estados con menos de N especies

        # Opciones de salida
        'percent': False,               # Similitud en 0-100 en vez de 0-1
        'export_distance': True,        # Exportar matriz de distancia (1 - similitud)
        'outdir': f'./resultados/upgma_{timestamp}',

        # Corte del dendrograma
        # Porcentaje de similitud donde se corta el árbol para definir grupos.
        # Ejemplo: 40 = agrupar todo lo que comparte >= 40% de similitud.
        # Poner None para omitir este paso.
        'umbral_corte': 40,
    }
    return configuracion


# ─────────────────────────────────────────────
#  VALIDACIÓN — no modificar
# ─────────────────────────────────────────────

EXTENSIONES_VALIDAS = {'.xlsx', '.xls', '.xlsm', '.xlsb'}

INDICES_VALIDOS = [
    'jaccard', 'simpson', 'sorensen', 'ochiai',
    'braun-blanquet', 'fager', 'kulezynski',
    'correlation', 'baroni'
]

METODOS_VALIDOS = ['single', 'complete', 'average']


def _error(mensaje):
    """Imprime un mensaje de error con formato y termina el programa."""
    print(f"\n{'='*60}")
    print(f"  ERROR: {mensaje}")
    print(f"{'='*60}\n")
    sys.exit(1)


def _advertencia(mensaje):
    """Imprime una advertencia sin terminar el programa."""
    print(f"  ADVERTENCIA: {mensaje}")


def validar_configuracion(config):
    """
    Verifica exhaustivamente que todos los parametros sean validos.
    Muestra mensajes de error claros antes de ejecutar el pipeline.
    """
    print("Validando configuracion...")
    errores = []
    advertencias = []

    # ── 1. Archivo de entrada ──────────────────────────────────────────

    archivo = Path(config.get('input_path', ''))

    if not config.get('input_path'):
        errores.append("'input_path' esta vacio. Debes indicar la ruta al archivo Excel.")

    else:
        # 1b. Que exista
        if not archivo.exists():
            errores.append(
                f"No se encontro el archivo: '{archivo}'\n"
                f"         Verifica que la ruta sea correcta y que el archivo exista."
            )

        else:
            # 1c. Extension valida
            if archivo.suffix.lower() not in EXTENSIONES_VALIDAS:
                errores.append(
                    f"Formato de archivo no soportado: '{archivo.suffix}'\n"
                    f"         Formatos validos: {', '.join(sorted(EXTENSIONES_VALIDAS))}\n"
                    f"         Si tu archivo es un CSV, conviertelo a Excel primero."
                )

            # 1d. Que sea un Excel real (no un archivo renombrado)
            elif archivo.suffix.lower() in {'.xlsx', '.xlsm', '.xlsb'}:
                try:
                    with zipfile.ZipFile(archivo, 'r'):
                        pass
                except zipfile.BadZipFile:
                    errores.append(
                        f"El archivo '{archivo}' tiene extension Excel pero su contenido "
                        f"no es valido.\n"
                        f"         Puede estar corrupto o ser un archivo renombrado."
                    )

            # 1e. Que no este vacio
            if archivo.exists() and archivo.stat().st_size == 0:
                errores.append(f"El archivo '{archivo}' esta vacio (0 bytes).")

    # ── 2. Indice de similitud ─────────────────────────────────────────

    sim_index = config.get('sim_index', '')
    if sim_index not in INDICES_VALIDOS:
        errores.append(
            f"Indice de similitud no valido: '{sim_index}'\n"
            f"         Opciones validas: {', '.join(INDICES_VALIDOS)}"
        )

    if sim_index in ['fager', 'kulezynski', 'correlation']:
        advertencias.append(
            f"El indice '{sim_index}' no esta acotado a [0,1] y puede producir\n"
            f"             distancias negativas o NaN en el dendrograma.\n"
            f"             Se recomienda usar: jaccard, sorensen u ochiai."
        )

    # ── 3. Metodo de linkage ───────────────────────────────────────────

    metodo = config.get('linkage_method', '')
    if metodo not in METODOS_VALIDOS:
        errores.append(
            f"Metodo de linkage no valido: '{metodo}'\n"
            f"         Opciones validas: {', '.join(METODOS_VALIDOS)}"
        )

    # ── 4. min_species ────────────────────────────────────────────────

    min_sp = config.get('min_species', 0)
    if not isinstance(min_sp, int) or min_sp < 0:
        errores.append(
            f"'min_species' debe ser un entero >= 0. Valor recibido: '{min_sp}'"
        )
    if isinstance(min_sp, int) and min_sp == 0:
        advertencias.append(
            "'min_species' es 0: se incluiran estados con muy pocas especies (ej. 'ND').\n"
            "             Se recomienda usar min_species=3 o mayor."
        )

    # ── 5. umbral_corte ───────────────────────────────────────────────

    umbral = config.get('umbral_corte')
    if umbral is not None:
        if not isinstance(umbral, (int, float)):
            errores.append(
                f"'umbral_corte' debe ser un numero entre 1 y 99. "
                f"Valor recibido: '{umbral}' ({type(umbral).__name__})"
            )
        elif not (0 < umbral < 100):
            errores.append(
                f"'umbral_corte' debe estar entre 1 y 99. Valor recibido: {umbral}"
            )

    # ── 6. percent ────────────────────────────────────────────────────

    if not isinstance(config.get('percent'), bool):
        errores.append(
            f"'percent' debe ser True o False. Valor recibido: '{config.get('percent')}'"
        )

    # ── 7. export_distance ────────────────────────────────────────────

    if not isinstance(config.get('export_distance'), bool):
        errores.append(
            f"'export_distance' debe ser True o False. "
            f"Valor recibido: '{config.get('export_distance')}'"
        )

    # ── 8. Directorio de salida ───────────────────────────────────────

    outdir = Path(config.get('outdir', ''))
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        errores.append(
            f"Sin permisos para crear el directorio de salida: '{outdir}'\n"
            f"         Elige otra ruta en 'outdir'."
        )
    except Exception as e:
        errores.append(f"No se pudo crear el directorio de salida '{outdir}': {e}")

    # ── Mostrar resultados ────────────────────────────────────────────

    if advertencias:
        print()
        for adv in advertencias:
            _advertencia(adv)

    if errores:
        print(f"\n{'='*60}")
        print(f"  Se encontraron {len(errores)} error(es). Corrige antes de continuar:")
        print(f"{'='*60}")
        for i, err in enumerate(errores, 1):
            print(f"\n  [{i}] {err}")
        print(f"\n{'='*60}\n")
        sys.exit(1)

    print("  Configuracion valida.\n")


# ─────────────────────────────────────────────
#  EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = cargar_configuracion()
    validar_configuracion(config)

    # ── Pipeline ──────────────────────────────────────────────────────
    try:
        resultados = pipeline(
            input_path=config['input_path'],
            sheet=config['sheet'],
            species_col=config['species_col'],
            state_col=config['state_col'],
            outdir=config['outdir'],
            min_species=config['min_species'],
            percent=config['percent'],
            export_distance=config['export_distance'],
            sim_index=config['sim_index'],
            linkage_method=config['linkage_method'],
            umbral_corte=config.get('umbral_corte'),
        )
    except ValueError as e:
        _error(
            f"Error al procesar los datos:\n         {e}\n\n"
            f"         Revisa que las columnas de especie y estado existan en el archivo,\n"
            f"         o especificalas manualmente con 'species_col' y 'state_col'."
        )
    except MemoryError:
        _error(
            "Memoria insuficiente para procesar el archivo.\n"
            "         Intenta filtrar mas datos con 'min_species' o usa un archivo mas pequeno."
        )
    except Exception as e:
        _error(f"Error inesperado durante el analisis:\n         {type(e).__name__}: {e}")

    # ── Validar resultados ────────────────────────────────────────────
    n_especies = resultados['pa'].shape[0]
    n_estados  = resultados['pa'].shape[1]

    if n_especies == 0:
        _error(
            "No se encontraron especies en los datos tras el filtrado.\n"
            "         Revisa el archivo de entrada o reduce el valor de 'min_species'."
        )
    if n_estados < 2:
        _error(
            f"Se necesitan al menos 2 estados para el clustering. "
            f"Solo se encontro {n_estados}.\n"
            f"         Revisa 'min_species' o el contenido del archivo."
        )

    # ── Resumen ───────────────────────────────────────────────────────
    print(f"  Resultados guardados en : {config['outdir']}")
    print(f"  Especies analizadas     : {n_especies}")
    print(f"  Estados analizados      : {n_estados}")
    print(f"  Indice                  : {config['sim_index']}")
    print(f"  Metodo                  : {config['linkage_method']}")
    print(f"  Umbral de corte         : {config['umbral_corte']}%"  if config.get('umbral_corte') is not None else f"  Umbral de corte         : No aplicado")

    # ── Corte del dendrograma ─────────────────────────────────────────
    umbral = config.get('umbral_corte')
    if umbral is not None and len(resultados.get('linkage', [])) > 0:
        print(f"\n--- Corte del dendrograma al {umbral}% de similitud ---")

        try:
            grupos = cortar_dendrograma(
                linkage=resultados['linkage'],
                labels=resultados['labels'],
                umbral_porcentaje=umbral,
                sim_index=config['sim_index'],
            )
        except Exception as e:
            _error(f"Error al cortar el dendrograma:\n         {type(e).__name__}: {e}")

        n_grupos = grupos['color'].nunique()

        if n_grupos == n_estados:
            _advertencia(
                f"Al {umbral}% cada estado quedo en su propio grupo ({n_grupos} grupos).\n"
                f"             Considera bajar 'umbral_corte' para obtener agrupaciones."
            )
        elif n_grupos == 1:
            _advertencia(
                f"Al {umbral}% todos los estados quedaron en un solo grupo.\n"
                f"             Considera subir 'umbral_corte' para obtener agrupaciones."
            )

        print(f"  Grupos formados: {n_grupos}\n")
        for g in sorted(grupos['color'].unique()):
            estados = grupos[grupos['color'] == g]['estado'].tolist()
            print(f"  Grupo {g:>2}: {estados}")

        # Guardar CSV
        try:
            outdir_str = config['outdir']
            os.makedirs(outdir_str, exist_ok=True)
            ruta_grupos = f"{outdir_str}/grupos_corte_{umbral}pct.csv"
            grupos.to_csv(ruta_grupos, index=False, encoding='utf-8-sig')
            print(f"\n  Asignacion guardada en: {ruta_grupos}")
        except PermissionError:
            _error(
                f"Sin permisos para escribir en '{outdir_str}'.\n"
                f"         Elige otra ruta en 'outdir'."
            )
        except Exception as e:
            _error(f"Error al guardar el CSV de grupos:\n         {type(e).__name__}: {e}")