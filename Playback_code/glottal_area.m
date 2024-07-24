function [vGLA]=glottal_area(vmAreas)
%[vGLA]=glottal_area(vmAreas)
% Calcula el área de la glotis de cada imagen y lo devuelve en un vector
%
% INPUTS:
%       - vmAreas: vector de matrices que incluye todas las imágenes de la
%       glotis. El bakground está a 0 y el área de la glotis, a 1.
%
% OUTPUTS: 
%       - vGLA: vector que contiene el área de la glotis en cada instante
%
% $Id:$

[~, ~,iNumImages]=size(vmAreas);
vGLA=zeros(1,iNumImages);
for i=1:iNumImages
    vGLA(i)=sum(sum(vmAreas(:,:,i)));
end
end
    
    
    