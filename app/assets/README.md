# Imágenes de ejemplo del aplicativo

Siete imágenes dermatoscópicas de HAM10000, una por clase, para probar el aplicativo sin
tener imágenes propias a mano. Son la **primera imagen de cada clase en el split de test**
(orden de `splits.csv`), elegidas sin mirar si el modelo acierta o no: ninguno de los dos
modelos las vio durante el entrenamiento ni la selección del checkpoint.

El nombre del archivo indica el diagnóstico de referencia y el identificador ISIC:
`<clase>_<image_id>.jpg`. El aplicativo muestra ese diagnóstico junto a la predicción.

## Fuente y licencia

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection
of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*,
5, 180161. Harvard Dataverse, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).

Las imágenes se distribuyen bajo licencia
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): uso no comercial con
atribución. Se incluyen sin modificaciones.
