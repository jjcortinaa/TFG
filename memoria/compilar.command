#!/bin/bash
# Doble clic sobre este fichero para compilar la memoria del TFG.
# Requiere tener MacTeX instalado (ver COMO_COMPILAR.md).

# Situarse en la carpeta donde está este script (la carpeta "memoria")
cd "$(dirname "$0")" || exit 1

# Asegurar que las herramientas de MacTeX están en el PATH
export PATH="/Library/TeX/texbin:/usr/local/bin:$PATH"

echo "=================================================="
echo "  Compilando TFG_spanish.tex ..."
echo "=================================================="
echo

latexmk -pdf -interaction=nonstopmode TFG_spanish.tex
RESULT=$?

echo
if [ $RESULT -eq 0 ] && [ -f TFG_spanish.pdf ]; then
    echo "OK  ->  PDF generado: TFG_spanish.pdf"
    open TFG_spanish.pdf
else
    echo "ERROR en la compilacion. Revisa el fichero TFG_spanish.log"
    echo "(busca la primera linea que empiece por '!')"
fi

echo
echo "Pulsa cualquier tecla para cerrar esta ventana."
read -n 1 -s
