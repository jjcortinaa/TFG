## Configuración de latexmk para la memoria del TFG
## Hace que `latexmk` compile la memoria con el número justo de pasadas
## (pdflatex -> biber -> pdflatex -> pdflatex) de forma automática.

$pdf_mode    = 1;          # genera el PDF con pdflatex
$bibtex_use  = 2;          # usa biber para la bibliografía (biblatex)
$biber       = 'biber %O %S';

# Ficheros intermedios que se borran con `latexmk -c`
$clean_ext   = 'bbl bcf run.xml fdb_latexmk fls synctex.gz nav snm';

# Vista previa: abre el PDF con el visor por defecto de macOS
$pdf_previewer = 'open %S';
