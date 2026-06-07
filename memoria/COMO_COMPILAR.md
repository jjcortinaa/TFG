# Cómo compilar la memoria del TFG en tu Mac

Overleaf gratuito da "compile timed out" porque tu memoria usa `biblatex` con
`biber`, lo que obliga a varias pasadas de compilación y supera el límite de
tiempo del plan gratuito. Compilando en local **no hay límite de tiempo** y
además va más rápido. Esto solo hay que configurarlo una vez.

---

## Paso 1 — Instalar Homebrew (si no lo tienes)

Abre la app **Terminal** y pega esto:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Si al escribir `brew --version` ya te aparece un número, ya lo tienes y puedes
saltarte este paso.

## Paso 2 — Instalar MacTeX

```
brew install --cask mactex-no-gui
```

Son unos 5 GB y tarda un rato. La versión `mactex-no-gui` trae todo lo
necesario (`pdflatex`, `biber`, `latexmk`, el idioma español de `babel`, los
paquetes `algorithm`, `biblatex`...) pero sin las apps gráficas que no
necesitas si vas a usar VS Code.

**Cierra la Terminal y vuelve a abrirla** cuando termine. Para comprobar que
está bien instalado:

```
which biber latexmk pdflatex
```

Deben aparecer tres rutas. Si no aparecen, reinicia el Mac y vuelve a probar.

---

## Paso 3 — Compilar. Tienes dos opciones

### Opción A (recomendada) — VS Code

1. Abre VS Code e instala la extensión **LaTeX Workshop** (de James Yu), en el
   panel de extensiones (el icono de los cuadraditos a la izquierda).
2. Abre la carpeta del TFG en VS Code (`Archivo > Abrir carpeta`).
3. Abre `memoria/TFG_spanish.tex`.
4. Guarda el fichero (`Cmd+S`). LaTeX Workshop compila solo y te abre el PDF
   en una vista previa al lado. A partir de ahí, cada vez que guardes se
   recompila.

LaTeX Workshop usa `latexmk` por debajo, así que ejecuta `biber` y las pasadas
necesarias automáticamente. No tienes que configurar nada más.

> Si VS Code se confunde de fichero principal (en la carpeta hay también
> `TFG_english.tex`), añade esta línea como **primera línea** de
> `TFG_spanish.tex`:
> `% !TEX root = TFG_spanish.tex`

### Opción B — Doble clic

En la carpeta `memoria` tienes el fichero **`compilar.command`**. Haz doble
clic encima y se compila solo en una ventana de Terminal; al terminar te abre
el PDF. (La primera vez macOS quizá pida confirmación: clic derecho > Abrir.)

### Opción C — Terminal a mano

```
cd ruta/a/la/carpeta/memoria
latexmk -pdf TFG_spanish.tex
```

La clave es usar `latexmk`, no `pdflatex` suelto: `latexmk` detecta que hay
`biber` y hace todas las pasadas en orden.

---

## Limpiar archivos intermedios

`latexmk` deja ficheros auxiliares (`.aux`, `.bbl`, `.bcf`, `.log`...). Para
borrarlos:

```
latexmk -c
```

---

## Si algo falla

- **Un paquete da error de "file not found"**: con `mactex-no-gui` completo es
  raro, pero si pasa, instálalo con `sudo tlmgr install nombre-del-paquete`.
- **Errores de compilación**: abre `TFG_spanish.log` y busca la **primera**
  línea que empiece por `!`. Ese es el error de verdad; lo de después suele ser
  consecuencia. Pásamelo y lo resolvemos.
- **La bibliografía sale vacía o con `[?]`**: falta la pasada de `biber`.
  Compila otra vez con `latexmk` (no con `pdflatex` solo) o borra con
  `latexmk -c` y recompila.

---

## Estado de la memoria (verificado)

Antes de escribir esta guía revisé `TFG_spanish.tex` y está estructuralmente
correcto: llaves balanceadas, entornos `\begin`/`\end` balanceados, las 39
claves `\cite` existen todas en los `.bib`, todas las referencias `\ref`
apuntan a un `\label` existente y las 4 imágenes existen. Debería compilar a la
primera en cuanto tengas MacTeX.
