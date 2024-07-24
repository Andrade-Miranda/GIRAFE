function [Imagen_normalizada]=Normalizacion_imagen(I,nivel)

Imagen_norm=(I-min(I(:))) ./ (max(I(:)-min(I(:))));

if (nivel~=1)
Imagen_normalizada=(round(Imagen_norm.*nivel));
else
Imagen_normalizada=Imagen_norm;
end